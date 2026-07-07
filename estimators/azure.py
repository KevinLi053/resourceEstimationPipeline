"""
Azure QDK Resource Estimator adapter.

Reuses estimation logic from:
  - estimator/circuitBuilderGeneralized.ipynb  (Steps 8–12)
  - estimator/analysis/hamlib.ipynb             (qdk.qre usage)

The public entry point is :func:`estimate`, which satisfies the
:class:`~estimators.base.Estimator` protocol.

Notes on metric availability
-----------------------------
Azure QDK performs T-gate synthesis and QEC distance selection internally;
several algorithmic gate counts (T count, Clifford count, rotation count,
toffoli count, T depth, logical depth) may not be directly exposed in the
result object depending on the qdk.qre version.  This adapter tries multiple
property name variants and records whatever is available; remaining fields are
left as None rather than silently returning N/A.

Properties that genuinely have no Azure equivalent:
  - rotation_synthesis_precision (rz_eps): Azure allocates an error budget
    fraction to rotation synthesis rather than accepting a precision directly.
    The closest proxy is NUM_TS_PER_ROTATION (already captured in t_per_rotation).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from qiskit import QuantumCircuit

from ..config import AzureConfig, PipelineConfig
from ..circuit.transpile import circuit_to_qasm, compute_t_depth
from .base import EstimationResult


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_factory(config: AzureConfig):
    """
    Construct the magic-state factory ISA query from config.

    Source: circuitBuilderGeneralized.ipynb — cell-qre-baseline.
    """
    from qdk.qre.models import RoundBasedFactory, Litinski19Factory

    if config.factory_type == "Litinski19":
        factory_cls = Litinski19Factory
    elif config.factory_type == "RoundBased":
        factory_cls = RoundBasedFactory

    if len(config.slow_down_factors) == 1 and config.slow_down_factors[0] == 1.0:
        return factory_cls.q()
    return factory_cls.q(slow_down_factor=config.slow_down_factors)


def _make_arch(config: AzureConfig):
    """
    Construct the GateBased hardware model from config.

    Source: circuitBuilderGeneralized.ipynb — cell-qre-baseline.
    """
    from qdk.qre.models import GateBased

    # GateBased.gate_time / measurement_time are typed as int and the
    # underlying Rust extension raises TypeError if floats are passed.
    # AzureConfig keeps them as float (nanoseconds can be non-integer in
    # principle), so we cast at the API boundary.
    return GateBased(
        error_rate=config.error_rate,
        gate_time=int(config.gate_time_ns),
        measurement_time=int(config.measurement_time_ns),
    )


def _extract_properties(best) -> dict:
    """
    Extract well-known properties from a single Pareto solution.

    Source: estimator/analysis/hamlib.ipynb — cell 866fe94b.
    """
    try:
        from qdk.qre import property_name
        return {property_name(k): v for k, v in best.properties.items()}
    except Exception:
        return {}


def _extract_t_info(best, config: AzureConfig) -> dict:
    """
    Extract T-gate space / time information from the best solution.

    Source: estimator/analysis/hamlib.ipynb — cell 866fe94b.
    """
    info: dict = {}
    try:
        from qdk.qre.instruction_ids import T
        if T in best.source:
            t_inst = best.source[T].instruction
            info["t_space"] = t_inst.space()
            info["t_time_ns"] = t_inst.time()
            info["t_error"] = t_inst.error_rate()
            # count() gives the number of T instructions scheduled in the circuit
            try:
                info["t_count"] = int(best.source[T].count)
            except Exception:
                pass
    except Exception:
        pass
    return info


def _get_prop(props: dict, *keys) -> Any:
    """
    Look up a property by trying multiple key name variants.

    Azure's property_name() output depends on the qdk.qre version; this helper
    tries several plausible names so the adapter remains robust across versions.
    Returns None if no key matches.
    """
    for k in keys:
        v = props.get(k)
        if v is not None:
            return v
    return None


def _extract_factory_summary(table_row) -> Optional[str]:
    """Try to get the factory summary string from an EstimationTable row."""
    try:
        return str(table_row.get("factories", ""))
    except Exception:
        return None


def _to_int_safe(v) -> Optional[int]:
    """Convert v to int, returning None on failure."""
    try:
        return int(v)
    except Exception:
        return None


def _to_float_safe(v) -> Optional[float]:
    """Convert v to float, returning None on failure."""
    try:
        return float(v)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public estimator function
# ---------------------------------------------------------------------------

def estimate(
    circuit: QuantumCircuit,
    config: PipelineConfig,
) -> EstimationResult:
    """
    Run the Microsoft QDK Resource Estimator on a Clifford+T circuit.

    Source: estimator/circuitBuilderGeneralized.ipynb — Steps 8–12.

    The circuit is exported to OpenQASM 3 and passed to ``qdk.qre.estimate()``.
    The Pareto-optimal solution at ``config.azure.pareto_index`` is returned
    as an :class:`EstimationResult`.

    Parameters
    ----------
    circuit : QuantumCircuit
        Canonical Clifford+T circuit (same one sent to the Qualtran estimator).
    config  : PipelineConfig

    Returns
    -------
    EstimationResult
    """
    from qdk.qre import estimate as _qre_estimate, EstimationTable
    from qdk.qre.application import OpenQASMApplication
    from qdk.qre.models import SurfaceCode

    az_cfg = config.azure

    # 1. Export circuit → OpenQASM 3
    qasm_source = circuit_to_qasm(circuit)

    # 2. Wrap as application
    app = OpenQASMApplication(qasm_source)

    # 3. Hardware model
    arch = _make_arch(az_cfg)

    # 4. ISA query
    isa_query = SurfaceCode.q() * _make_factory(az_cfg)

    # 5. Run
    result = _qre_estimate(
        app,
        arch,
        isa_query,
        max_error=az_cfg.error_budget,
    )

    if len(result) == 0:
        raise RuntimeError("Azure QDK returned no Pareto-optimal solutions.")

    # 6. Pick the requested Pareto solution
    idx = az_cfg.pareto_index % len(result)
    best = result[idx]

    # 7. Extract properties
    props = _extract_properties(best)
    t_info = _extract_t_info(best, az_cfg)

    # Compute circuit-derived T depth before any property lookups.
    # This is a property of the transpiled circuit, not of the estimator,
    # and is used as a fallback when Azure does not expose t_depth.
    circuit_t_depth = compute_t_depth(circuit)

    # Try to get factory summary via EstimationTable
    factory_str: Optional[str] = None
    try:
        table = EstimationTable()
        table.extend(result)
        table.add_qubit_partition_column()
        table.add_factory_summary_column()
        df = table.as_frame()
        if "factories" in df.columns:
            factory_str = str(df.iloc[idx]["factories"])
    except Exception:
        pass

    # 8. Map to EstimationResult
    df_raw = result.as_frame()
    row = df_raw.iloc[idx]

    # ── Physical qubits ───────────────────────────────────────────────────────
    phys_qubits = int(row.get("qubits", props.get("ALGORITHM_COMPUTE_QUBITS", None) or 0))
    compute = _to_int_safe(_get_prop(props,
        "PHYSICAL_COMPUTE_QUBITS", "physicalQubitsForAlgorithm", "COMPUTE_QUBITS"))
    factory = _to_int_safe(_get_prop(props,
        "PHYSICAL_FACTORY_QUBITS", "physicalQubitsForTfactories", "FACTORY_QUBITS"))
    memory  = _to_int_safe(_get_prop(props,
        "PHYSICAL_MEMORY_QUBITS", "MEMORY_QUBITS"))
    logical = _to_int_safe(_get_prop(props,
        "ALGORITHM_COMPUTE_QUBITS", "algorithmicLogicalQubits", "LOGICAL_QUBITS"))
    logical_compute = _get_prop(props, "LOGICAL_COMPUTE_QUBITS", "logicalQubits")

    total_phys = (
        (compute or 0) + (factory or 0) + (memory or 0)
    ) or phys_qubits

    # ── Runtime ───────────────────────────────────────────────────────────────
    runtime_td = row.get("runtime")
    runtime_s: Optional[float] = None
    if runtime_td is not None:
        try:
            runtime_s = runtime_td.total_seconds()
        except Exception:
            try:
                runtime_s = float(runtime_td) * 1e-9
            except Exception:
                pass

    # ── Logical error rate ────────────────────────────────────────────────────
    error = row.get("error")
    error_f = _to_float_safe(error)

    # ── Code distance ─────────────────────────────────────────────────────────
    # Azure selects a surface-code distance internally during optimisation.
    # The chosen distance may be reported under several property names depending
    # on the qdk.qre version.  We try:
    #   1. well-known property name variants from best.properties (via props dict)
    #   2. columns on the raw Pareto DataFrame (df_raw)
    #   3. direct attributes on the best solution object
    # To discover all available keys for a given qdk.qre version, inspect
    # result.extra["props"] after a run (all properties are stored there).
    code_dist = _to_int_safe(_get_prop(props,
        "CODE_DISTANCE", "codeDistance", "PHYSICAL_QUBIT_CODE_DISTANCE",
        "DATA_CODE_DISTANCE", "dataCodeDistance",
        "qubitParams.codeDistance", "qubitParams_codeDistance",
        "SURFACE_CODE_DISTANCE", "surfaceCodeDistance"))

    if code_dist is None:
        # Fall back to the Pareto DataFrame row (some versions include it as a column)
        for col in ("codeDistance", "code_distance", "dataCodeDistance",
                    "CODE_DISTANCE", "surfaceCodeDistance"):
            v = _to_int_safe(row.get(col))
            if v is not None:
                code_dist = v
                break

    if code_dist is None:
        # Last resort: direct attribute on the best solution object
        for attr in ("code_distance", "codeDistance", "data_code_distance"):
            try:
                v = _to_int_safe(getattr(best, attr, None))
                if v is not None:
                    code_dist = v
                    break
            except Exception:
                pass

    # ── Logical cycles ────────────────────────────────────────────────────────
    # Number of QEC rounds for the full algorithm (not always exposed).
    logical_cycles = _to_int_safe(_get_prop(props,
        "NUM_LOGICAL_CYCLES", "numCycles", "LOGICAL_CYCLES", "logicalCycles",
        "NUM_CYCLES", "ALGORITHM_CYCLES"))

    # ── Logical depth ─────────────────────────────────────────────────────────
    # Gate-layer depth of the logical circuit (not always exposed by QDK).
    logical_depth = _to_int_safe(_get_prop(props,
        "ALGORITHM_DEPTH", "algorithmicLogicalDepth", "LOGICAL_DEPTH",
        "CIRCUIT_DEPTH", "circuitDepth"))

    # ── T count ───────────────────────────────────────────────────────────────
    # Azure does not always report T count directly; try instruction count first,
    # then fall back to properties.  Approximate: Azure synthesises Rz internally
    # so the T count it uses may differ from a pure Clifford+T input count.
    t_count = t_info.get("t_count")
    if t_count is None:
        t_count = _to_int_safe(_get_prop(props,
            "NUM_T_STATES", "numTs", "T_COUNT", "ALGORITHM_T_COUNT",
            "tCount", "NUM_TS"))

    # ── T depth ───────────────────────────────────────────────────────────────
    # Prefer the estimator-reported value; fall back to the circuit-derived
    # T depth (DAG layer analysis) which is always available.
    t_depth = _to_int_safe(_get_prop(props,
        "ALGORITHM_T_DEPTH", "tDepth", "T_DEPTH", "algorithmicTDepth"))
    if t_depth is None:
        t_depth = circuit_t_depth  # derived: T-gate layer count from input circuit

    # ── Clifford count ────────────────────────────────────────────────────────
    # Cliffords are generally free in fault-tolerant computing (absorbed into
    # the syndrome schedule), so QDK may not report this.
    clifford_count = _to_int_safe(_get_prop(props,
        "ALGORITHM_CLIFFORD_COUNT", "cliffordCount", "CLIFFORD_COUNT",
        "algorithmicCliffordCount"))

    # ── Rotation count ────────────────────────────────────────────────────────
    rotation_count = _to_int_safe(_get_prop(props,
        "ALGORITHM_ROTATION_COUNT", "rotationCount", "ROTATION_COUNT",
        "algorithmicRotationCount", "NUM_ROTATIONS"))

    # ── Toffoli count ─────────────────────────────────────────────────────────
    toffoli_count = _to_int_safe(_get_prop(props,
        "ALGORITHM_TOFFOLI_COUNT", "toffoliCount", "TOFFOLI_COUNT",
        "algorithmicToffoliCount", "NUM_TOFFOLIS", "CCZ_COUNT"))

    # ── Measurement count ─────────────────────────────────────────────────────
    measurement_count = _to_int_safe(_get_prop(props,
        "ALGORITHM_MEASUREMENT_COUNT", "measurementCount", "MEASUREMENT_COUNT",
        "algorithmicMeasurementCount", "NUM_MEASUREMENTS"))

    # ── T gates per rotation ──────────────────────────────────────────────────
    t_per_rot = _get_prop(props,
        "NUM_TS_PER_ROTATION", "numTsPerRotation", "T_GATES_PER_ROTATION",
        "tGatesPerRotation")
    t_per_rot_int = _to_int_safe(t_per_rot)

    # ── Logical cycle time → derive logical_cycles from runtime if not in props ─
    # DERIVED: logical_cycles = runtime_s / logical_cycle_time_s
    # Logical cycle time is either read directly from props or approximated as
    #   2 * code_distance * measurement_time_ns  (dominant term in GateBased model).
    if logical_cycles is None and runtime_s is not None:
        # Prefer the estimator-reported cycle time if available
        logical_cycle_time_ns_raw = _to_float_safe(_get_prop(props,
            "LOGICAL_CYCLE_TIME", "logicalCycleTime", "CYCLE_TIME_NS",
            "logicalCycleTimeNs", "LOGICAL_CYCLE_TIME_NS"))
        if logical_cycle_time_ns_raw is not None and logical_cycle_time_ns_raw > 0:
            # derived from Azure-reported cycle time and runtime
            logical_cycles = int(runtime_s * 1e9 / logical_cycle_time_ns_raw)
        elif code_dist is not None:
            # derived (approximate): one surface-code round ≈ 2*d × meas_time_ns
            approx_cycle_ns = 2 * code_dist * az_cfg.measurement_time_ns
            if approx_cycle_ns > 0:
                logical_cycles = int(runtime_s * 1e9 / approx_cycle_ns)

    # ── Factory instances ─────────────────────────────────────────────────────
    factory_instances = _to_int_safe(_get_prop(props,
        "NUM_TFACTORIES", "numTfactories", "NUM_FACTORIES", "FACTORY_COUNT",
        "tfactories"))

    return EstimationResult(
        estimator_name=f"Azure QDK (err_rate={az_cfg.error_rate:.0e}, budget={az_cfg.error_budget})",
        logical_qubits=logical,
        logical_depth=logical_depth,
        logical_cycles=logical_cycles,
        t_count=t_count,
        t_depth=t_depth,
        clifford_count=clifford_count,
        rotation_count=rotation_count,
        toffoli_count=toffoli_count,
        measurement_count=measurement_count,
        physical_qubits=total_phys,
        physical_compute_qubits=compute,
        physical_factory_qubits=factory,
        physical_memory_qubits=memory,
        runtime_seconds=runtime_s,
        error_budget=az_cfg.error_budget,
        logical_error_rate=error_f,
        code_distance=code_dist,
        factory_type=az_cfg.factory_type,
        factory_count=factory_str,
        t_per_rotation=t_per_rot_int,
        # Azure does not accept rz_eps directly; rotation synthesis precision is
        # controlled via the error budget allocation.  No direct equivalent.
        rotation_synthesis_precision=None,
        physical_error_rate=az_cfg.error_rate,
        gate_time_ns=az_cfg.gate_time_ns,
        measurement_time_ns=az_cfg.measurement_time_ns,
        # Azure does not use a cycle_time_us parameter (it uses gate_time_ns instead).
        cycle_time_us=None,
        algorithm_assumptions=(
            f"Clifford+T circuit; basis={config.transpile.basis_gates}; "
            f"PauliEvolutionGate SuzukiTrotter order={config.evolution.synthesis_order} "
            f"reps={config.evolution.synthesis_reps}; "
            f"t={config.evolution.evolution_time}"
        ),
        architecture_assumptions=(
            f"GateBased error_rate={az_cfg.error_rate}, "
            f"gate_time={az_cfg.gate_time_ns} ns, "
            f"meas_time={az_cfg.measurement_time_ns} ns; "
            f"SurfaceCode QEC; factory={az_cfg.factory_type}"
        ),
        raw=result,
        extra={
            "all_pareto_solutions": df_raw.to_dict("records"),
            "t_instruction_space": t_info.get("t_space"),
            "t_instruction_time_ns": t_info.get("t_time_ns"),
            "t_instruction_error": t_info.get("t_error"),
            "props": props,
            "logical_compute_qubits": logical_compute,
            # factory_instances is not exposed in the unified schema but stored here
            "factory_instances": factory_instances,
        },
    )
