"""
Azure QDK Resource Estimator adapter.

Reuses estimation logic from:
  - estimator/circuitBuilderGeneralized.ipynb  (Steps 8–12)
  - estimator/analysis/hamlib.ipynb             (qdk.qre usage)

The public entry point is :func:`estimate`, which satisfies the
:class:`~resourceEstimationPipeline.estimators.base.Estimator` protocol.
"""
from __future__ import annotations

from typing import Any, Optional

from qiskit import QuantumCircuit

from resourceEstimationPipeline.config import AzureConfig, PipelineConfig
from resourceEstimationPipeline.circuit.transpile import circuit_to_qasm
from resourceEstimationPipeline.estimators.base import EstimationResult


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
    except Exception:
        pass
    return info


def _extract_factory_summary(table_row) -> Optional[str]:
    """Try to get the factory summary string from an EstimationTable row."""
    try:
        return str(table_row.get("factories", ""))
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

    phys_qubits = int(row.get("qubits", props.get("ALGORITHM_COMPUTE_QUBITS", None) or 0))
    # More reliable total qubits from properties
    compute = props.get("PHYSICAL_COMPUTE_QUBITS")
    factory = props.get("PHYSICAL_FACTORY_QUBITS")
    memory  = props.get("PHYSICAL_MEMORY_QUBITS")
    logical = props.get("ALGORITHM_COMPUTE_QUBITS")
    logical_compute = props.get("LOGICAL_COMPUTE_QUBITS")

    total_phys = (
        (compute or 0) + (factory or 0) + (memory or 0)
    ) or phys_qubits

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

    error = row.get("error")
    try:
        error_f = float(error)
    except Exception:
        error_f = None

    t_per_rot = props.get("NUM_TS_PER_ROTATION")

    return EstimationResult(
        estimator_name=f"Azure QDK (err_rate={az_cfg.error_rate:.0e}, budget={az_cfg.error_budget})",
        logical_qubits=logical,
        t_count=None,       # QDK does not directly report T count
        t_depth=None,
        clifford_count=None,
        rotation_count=None,
        toffoli_count=None,
        measurement_count=None,
        physical_qubits=total_phys,
        physical_compute_qubits=compute,
        physical_factory_qubits=factory,
        physical_memory_qubits=memory,
        runtime_seconds=runtime_s,
        error_budget=az_cfg.error_budget,
        logical_error_rate=error_f,
        code_distance=None,  # Azure QDK sweeps distances internally; not directly exposed
        factory_count=factory_str,
        t_per_rotation=int(t_per_rot) if t_per_rot is not None else None,
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
        },
    )
