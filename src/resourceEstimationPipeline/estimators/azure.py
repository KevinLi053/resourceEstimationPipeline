"""
Azure QDK Resource Estimator adapter.

Reuses estimation logic from:
  - estimator/circuitBuilderGeneralized.ipynb  (Steps 8–12)
  - estimator/analysis/hamlib.ipynb             (qdk.qre usage)

The public entry point is :func:`estimate`, which satisfies the
:class:`~estimators.base.Estimator` protocol.

━━━ How qdk.qre stores results ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After `qre.estimate()` the adapter receives an `EstimationTable` of
`EstimationTableEntry` objects.  Each entry exposes:

  entry.qubits    – total physical qubits (int)
  entry.runtime   – total runtime in nanoseconds (int)
  entry.error     – total logical error probability (float)

  entry.properties – dict[int, int|float|bool|str]
      Keys are integer constants from `qdk.qre.property_keys`:
        PHYSICAL_COMPUTE_QUBITS  (10) – physical qubits for the compute register
        PHYSICAL_FACTORY_QUBITS  (11) – physical qubits for the T factory
        PHYSICAL_MEMORY_QUBITS   (12) – physical qubits for logical memory (may be absent)
        LOGICAL_COMPUTE_QUBITS   (14) – QEC-encoded logical compute qubits
        LOGICAL_MEMORY_QUBITS    (15) – QEC-encoded logical memory qubits
        ALGORITHM_COMPUTE_QUBITS (16) – algorithmic (circuit) qubit count
        ALGORITHM_MEMORY_QUBITS  (17) – algorithmic memory qubit count
        NUM_TS_PER_ROTATION      ( 6) – T gates used per arbitrary Rz rotation
        EVALUATION_TIME          ( 9) – evaluation overhead time in ns
        RUNTIME_SINGLE_SHOT      ( 7) – single-shot runtime in ns (if set)

  entry.factories – dict[int, FactoryResult]  (keyed by instruction ID)
      Key is the instruction ID of the magic-state factory (e.g., T = 1028).
      FactoryResult has:
        .copies  – parallel factory instances (= num_factories)
        .runs    – factory invocations by the algorithm (T-state demand proxy)
        .states  – T states produced per factory run
        .error_rate – factory output error rate

  entry.source – InstructionSource graph
      Contains the ISA instructions used for estimation.  The LATTICE_SURGERY
      instruction (id=4352) carries QEC parameters:
        .instruction.get_property(DISTANCE)        – code distance d
        .instruction.get_property(CODE_CYCLE_TIME) – syndrome-extraction cycle time (ns)
        .instruction.time(arity=1)                 – logical cycle time = d × cycle_time (ns)
        .transform.distance                        – same distance, via the SurfaceCode object

━━━ What QRE does NOT expose ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  - logical_depth   : circuit gate-layer depth — not tracked post-PSSPC.
  - t_depth         : T-gate critical path — not reported; use circuit analysis.
  - t_count         : raw circuit T count — PSSPC transforms the circuit before
                      estimation; factory.runs reflects T-state demands from the
                      PSSPC-transformed trace, not the raw gate count.
  - clifford_count  : not tracked (Cliffords are free in fault-tolerant computing).
  - rotation_count  : total Rz count — only NUM_TS_PER_ROTATION (per-rotation rate)
                      is reported; total rotation count must come from circuit analysis.
  - toffoli_count   : not reported.
  - measurement_count: not reported.
  - rotation_synthesis_precision (rz_eps): Azure uses an error-budget fraction for
                      rotation synthesis rather than a fixed precision.  The closest
                      proxy is NUM_TS_PER_ROTATION (captured in t_per_rotation).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from qiskit import QuantumCircuit

from ..config import AzureConfig, PipelineConfig
from ..circuit.transpile import circuit_to_qasm, compute_t_depth
from .base import EstimationResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_factory(config: AzureConfig):
    """Construct the magic-state factory ISA query from config."""
    from qdk.qre.models import RoundBasedFactory, Litinski19Factory

    factory_cls = Litinski19Factory if config.factory_type == "Litinski19" else RoundBasedFactory

    if len(config.slow_down_factors) == 1 and config.slow_down_factors[0] == 1.0:
        return factory_cls.q()
    return factory_cls.q(slow_down_factor=config.slow_down_factors)


def _make_arch(config: AzureConfig):
    """Construct the GateBased hardware model from config."""
    from qdk.qre.models import GateBased

    two_qubit_gate_time = (
        config.two_qubit_gate_time_ns
        if config.two_qubit_gate_time_ns is not None
        else config.gate_time_ns
    )
    # GateBased requires int times; cast here at the API boundary.
    return GateBased(
        error_rate=config.error_rate,
        gate_time=int(config.gate_time_ns),
        measurement_time=int(config.measurement_time_ns),
        two_qubit_gate_time=int(two_qubit_gate_time),
    )


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


def _extract_qec_params(best) -> dict:
    """
    Extract QEC code parameters from the LATTICE_SURGERY instruction.

    The SurfaceCode ISA transform stores the code distance as a property on
    the LATTICE_SURGERY instruction node (key = DISTANCE = 0) and the syndrome
    extraction cycle time as CODE_CYCLE_TIME (key = 21).  The logical cycle
    time equals d × code_cycle_time = instruction.time(arity=1).

    Returns a dict with keys (all may be absent if extraction fails):
      "code_distance"       – int
      "code_cycle_time_ns"  – int, syndrome extraction round time in ns
      "logical_cycle_time_ns" – int, = d × code_cycle_time in ns
    """
    from qdk.qre.instruction_ids import LATTICE_SURGERY
    from qdk.qre.property_keys import DISTANCE, CODE_CYCLE_TIME

    result: dict = {}
    try:
        ls_ref = best.source.get(LATTICE_SURGERY)
        if ls_ref is None:
            log.debug("_extract_qec_params: LATTICE_SURGERY not in source graph")
            return result

        # Dereference the node to get the raw Instruction object
        ls_node = best.source.nodes[ls_ref.node_id]
        ls_instr = ls_node.instruction

        # Code distance: stored as DISTANCE property on the instruction by SurfaceCode
        distance = ls_instr.get_property(DISTANCE)
        if distance is not None:
            result["code_distance"] = int(distance)
        elif hasattr(ls_ref, "transform") and hasattr(ls_ref.transform, "distance"):
            # Fallback: read from the SurfaceCode transform object directly
            result["code_distance"] = int(ls_ref.transform.distance)

        # Syndrome extraction cycle time in ns (stored as CODE_CYCLE_TIME)
        cct = ls_instr.get_property(CODE_CYCLE_TIME)
        if cct is not None:
            result["code_cycle_time_ns"] = int(cct)

        # Logical cycle time = d × code_cycle_time = instruction time per 1-qubit op
        lct = ls_instr.time(1)
        if lct is not None:
            result["logical_cycle_time_ns"] = int(lct)

        log.debug(
            "_extract_qec_params: d=%s  code_cycle_ns=%s  logical_cycle_ns=%s",
            result.get("code_distance"),
            result.get("code_cycle_time_ns"),
            result.get("logical_cycle_time_ns"),
        )
    except Exception as exc:
        log.debug("_extract_qec_params failed: %s", exc)

    return result


def _extract_factory(best) -> dict:
    """
    Extract T-factory metrics from best.factories.

    best.factories is dict[instruction_id, FactoryResult].  For the
    standard Litinski19Factory / RoundBasedFactory the magic-state
    instruction ID is T (= 1028).

    FactoryResult attributes:
      .copies  – parallel factory instances          → num_factories
      .runs    – sequential factory invocations      → T-state demand proxy
      .states  – T states produced per invocation
      .error_rate – output T-state error rate

    Note: .copies × .runs × .states gives the total T states produced, which
    accounts for PSSPC lattice-surgery overhead and may exceed the raw circuit
    T gate count significantly.

    Returns a dict with keys (all may be absent):
      "num_factories"         – int, parallel factories (.copies)
      "factory_runs"          – int (.runs)
      "factory_states_per_run"– int (.states)
      "factory_error_rate"    – float
      "t_state_count"         – int, total T states produced (copies×runs×states)
      "factory_summary"       – str, e.g. "2×T"
    """
    from qdk.qre.instruction_ids import T
    from qdk.qre._qre import instruction_name

    result: dict = {}
    try:
        if best.factories:
            parts = [
                f"{fr.copies}×{instruction_name(fid) or str(fid)}"
                for fid, fr in best.factories.items()
            ]
            result["factory_summary"] = ", ".join(parts)
        else:
            result["factory_summary"] = "None"

        # T-factory (covers Litinski19Factory and RoundBasedFactory output)
        t_fr = best.factories.get(T)
        if t_fr is not None:
            result["num_factories"] = int(t_fr.copies)
            result["factory_runs"] = int(t_fr.runs)
            result["factory_states_per_run"] = int(t_fr.states)
            result["factory_error_rate"] = float(t_fr.error_rate)
            result["t_state_count"] = int(t_fr.copies) * int(t_fr.runs) * int(t_fr.states)

        log.debug(
            "_extract_factory: %s  num_factories=%s  t_states=%s",
            result.get("factory_summary"),
            result.get("num_factories"),
            result.get("t_state_count"),
        )
    except Exception as exc:
        log.debug("_extract_factory failed: %s", exc)

    return result


def _log_diagnostic(best, props: dict, qec: dict, factory: dict) -> None:
    """Emit DEBUG-level diagnostic summary of all extracted QRE values."""
    if not log.isEnabledFor(logging.DEBUG):
        return

    from qdk.qre._qre import property_name

    log.debug("── Azure QRE Diagnostic ──────────────────────────────────")
    log.debug("  entry.qubits   = %s  (total physical)", best.qubits)
    log.debug("  entry.runtime  = %s ns", best.runtime)
    log.debug("  entry.error    = %s", best.error)

    log.debug("  entry.properties (%d keys):", len(props))
    for k, v in sorted(props.items()):
        pname = property_name(k)
        log.debug("    key=%d  name=%r  value=%r", k, pname, v)

    log.debug("  QEC params: %s", qec)
    log.debug("  Factory:    %s", factory)

    if not best.factories:
        log.debug("  entry.factories: (empty)")
    else:
        for fid, fr in best.factories.items():
            from qdk.qre._qre import instruction_name
            log.debug(
                "  factory id=%d (%s): copies=%d  runs=%d  states=%d  error=%.3e",
                fid, instruction_name(fid), fr.copies, fr.runs, fr.states, fr.error_rate,
            )

    log.debug("─────────────────────────────────────────────────────────")


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
    from qdk.qre import estimate as _qre_estimate
    from qdk.qre.application import OpenQASMApplication
    from qdk.qre.models import SurfaceCode
    from qdk.qre.property_keys import (
        PHYSICAL_COMPUTE_QUBITS, PHYSICAL_FACTORY_QUBITS, PHYSICAL_MEMORY_QUBITS,
        ALGORITHM_COMPUTE_QUBITS, ALGORITHM_MEMORY_QUBITS,
        LOGICAL_COMPUTE_QUBITS, LOGICAL_MEMORY_QUBITS,
        NUM_TS_PER_ROTATION, EVALUATION_TIME, RUNTIME_SINGLE_SHOT,
    )
    from qdk.qre._trace import PSSPC, LatticeSurgery

    az_cfg = config.azure

    # 1. Export circuit → OpenQASM 3
    qasm_source = circuit_to_qasm(circuit)

    # 2. Wrap as application
    app = OpenQASMApplication(qasm_source)

    # 3. Hardware model
    arch = _make_arch(az_cfg)

    # 4. ISA query — fix distance when specified, otherwise sweep the default domain
    sc_kwargs: dict = {}
    if az_cfg.code_distance is not None:
        sc_kwargs["distance"] = az_cfg.code_distance
    isa_query = SurfaceCode.q(**sc_kwargs) * _make_factory(az_cfg)

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
    best = result[idx]  # EstimationTableEntry

    # ── Native property access (integer keys — authoritative) ─────────────────
    # best.properties is dict[int, bool|int|float|str]; use named constants.
    props = best.properties  # direct reference, not a copy

    # Physical qubit partition
    phys_compute = _to_int_safe(props.get(PHYSICAL_COMPUTE_QUBITS))
    phys_factory = _to_int_safe(props.get(PHYSICAL_FACTORY_QUBITS))
    phys_memory  = _to_int_safe(props.get(PHYSICAL_MEMORY_QUBITS))

    # Algorithmic and QEC-encoded qubit counts
    # ALGORITHM_COMPUTE_QUBITS = circuit qubit count (width of the input circuit)
    # LOGICAL_COMPUTE_QUBITS   = QEC-encoded logical qubits (includes ancilla for routing)
    algo_compute = _to_int_safe(props.get(ALGORITHM_COMPUTE_QUBITS))
    algo_memory  = _to_int_safe(props.get(ALGORITHM_MEMORY_QUBITS))
    log_compute  = _to_int_safe(props.get(LOGICAL_COMPUTE_QUBITS))
    log_memory   = _to_int_safe(props.get(LOGICAL_MEMORY_QUBITS))

    # T-gate synthesis rate per arbitrary Rz rotation (set by PSSPC transform)
    t_per_rot = _to_int_safe(props.get(NUM_TS_PER_ROTATION))

    # Miscellaneous timing properties
    eval_time_ns  = _to_int_safe(props.get(EVALUATION_TIME))
    single_shot_ns = _to_int_safe(props.get(RUNTIME_SINGLE_SHOT))

    # ── Top-level EstimationTableEntry fields (definitive) ────────────────────
    total_phys  = best.qubits           # total physical qubits (int)
    runtime_ns  = best.runtime          # total runtime in nanoseconds (int)
    error_rate  = best.error            # total logical error probability (float)
    runtime_s   = runtime_ns / 1e9 if runtime_ns else None

    # ── QEC parameters from the LATTICE_SURGERY instruction ───────────────────
    # Code distance and cycle times are NOT in best.properties; they are stored
    # as properties on the LATTICE_SURGERY instruction in the ISA source graph.
    qec_params = _extract_qec_params(best)
    code_dist             = qec_params.get("code_distance")
    code_cycle_time_ns    = qec_params.get("code_cycle_time_ns")
    logical_cycle_time_ns = qec_params.get("logical_cycle_time_ns")

    # Fallback: config specifies a fixed distance → always report it even if
    # extraction fails (e.g. when use_graph=True skips the ISA rebuild).
    if code_dist is None and az_cfg.code_distance is not None:
        code_dist = az_cfg.code_distance

    # ── Logical cycles ────────────────────────────────────────────────────────
    # Derived from total runtime divided by one logical cycle time.
    # logical_cycle_time_ns = d × code_cycle_time_ns (time per LATTICE_SURGERY op).
    logical_cycles: Optional[int] = None
    if runtime_ns and logical_cycle_time_ns and logical_cycle_time_ns > 0:
        logical_cycles = runtime_ns // logical_cycle_time_ns

    # ── Factory metrics from best.factories ───────────────────────────────────
    # num_factories = parallel factory copies (optimizer-chosen).
    # The factory's T-state demand (factory_runs × states) reflects the
    # PSSPC-transformed trace, not the raw circuit T count.
    factory_info    = _extract_factory(best)
    num_factories   = factory_info.get("num_factories")
    factory_summary = factory_info.get("factory_summary", "None")

    # ── Circuit-derived gate counts ───────────────────────────────────────────
    # QRE does not expose the raw circuit T count, logical depth, T depth,
    # Clifford count, rotation count, Toffoli count, or measurement count.
    # These fields are left None here and populated by enrich_from_circuit()
    # in compare/metrics.py using Qiskit circuit analysis.
    circuit_t_depth = compute_t_depth(circuit)

    # ── Diagnostic logging ────────────────────────────────────────────────────
    _log_diagnostic(best, props, qec_params, factory_info)
    log.debug(
        "Azure summary: phys=%s (compute=%s factory=%s memory=%s)  "
        "algo_qubits=%s  d=%s  logical_cycles=%s  num_factories=%s  "
        "runtime_ns=%s  error=%.4g",
        total_phys, phys_compute, phys_factory, phys_memory,
        algo_compute, code_dist, logical_cycles, num_factories,
        runtime_ns, error_rate,
    )

    return EstimationResult(
        estimator_name=(
            f"Azure QDK (err_rate={az_cfg.error_rate:.0e}, "
            f"budget={az_cfg.error_budget})"
        ),

        # ── Logical / algorithmic ─────────────────────────────────────────────
        # logical_qubits = ALGORITHM_COMPUTE_QUBITS: circuit width as given to QRE
        logical_qubits=algo_compute,
        # logical_depth, t_count, t_depth, clifford_count, rotation_count,
        # toffoli_count, measurement_count → left None; enrich_from_circuit() fills them.
        logical_depth=None,
        logical_cycles=logical_cycles,
        t_count=None,
        t_depth=circuit_t_depth,   # circuit DAG T-depth (available pre-estimation)
        clifford_count=None,
        rotation_count=None,
        toffoli_count=None,
        measurement_count=None,

        # ── Physical resources ────────────────────────────────────────────────
        # physical_qubits = best.qubits (total, authoritative)
        physical_qubits=total_phys,
        physical_compute_qubits=phys_compute,
        physical_factory_qubits=phys_factory,
        physical_memory_qubits=phys_memory,

        # ── Timing ───────────────────────────────────────────────────────────
        runtime_seconds=runtime_s,

        # ── Error budget & QEC ────────────────────────────────────────────────
        error_budget=az_cfg.error_budget,
        # logical_error_rate = best.error (total logical failure probability)
        logical_error_rate=error_rate,
        code_distance=code_dist,
        # New QEC timing fields
        logical_cycle_time_ns=logical_cycle_time_ns,
        code_cycle_time_ns=code_cycle_time_ns,

        # ── Factory ──────────────────────────────────────────────────────────
        factory_type=az_cfg.factory_type,
        factory_count=factory_summary,
        # num_factories = best.factories[T].copies (optimizer-chosen parallel count)
        num_factories=num_factories,

        # ── Rotation synthesis ────────────────────────────────────────────────
        # t_per_rotation = NUM_TS_PER_ROTATION (T gates per Rz synthesised by PSSPC)
        t_per_rotation=t_per_rot,
        # rotation_synthesis_precision: Azure uses budget-fraction allocation, not ε.
        rotation_synthesis_precision=None,

        # ── Physical hardware parameters ──────────────────────────────────────
        physical_error_rate=az_cfg.error_rate,
        gate_time_ns=az_cfg.gate_time_ns,
        measurement_time_ns=az_cfg.measurement_time_ns,
        cycle_time_us=None,  # Azure uses gate_time_ns + meas_time_ns instead

        # ── Assumptions ──────────────────────────────────────────────────────
        algorithm_assumptions=(
            f"Clifford+T circuit; basis={config.transpile.basis_gates}; "
            f"PauliEvolutionGate SuzukiTrotter order={config.evolution.synthesis_order} "
            f"reps={config.evolution.synthesis_reps}; "
            f"t={config.evolution.evolution_time}"
        ),
        architecture_assumptions=(
            f"GateBased error_rate={az_cfg.error_rate:.0e}; "
            f"gate_time={az_cfg.gate_time_ns} ns; "
            f"meas_time={az_cfg.measurement_time_ns} ns; "
            f"SurfaceCode QEC d={code_dist}; "
            f"factory={az_cfg.factory_type}"
        ),

        # ── Raw / extra ───────────────────────────────────────────────────────
        raw=result,
        extra={
            # All Pareto-optimal solutions from this run
            "all_pareto_solutions": result.as_frame().to_dict("records"),
            # QEC-encoded logical qubit counts (includes routing ancilla)
            "logical_compute_qubits": log_compute,
            "logical_memory_qubits": log_memory,
            # Algorithmic memory qubits
            "algorithm_memory_qubits": algo_memory,
            # Timing detail
            "evaluation_time_ns": eval_time_ns,
            "single_shot_runtime_ns": single_shot_ns,
            # Factory detail
            "factory_runs": factory_info.get("factory_runs"),
            "factory_states_per_run": factory_info.get("factory_states_per_run"),
            "factory_error_rate": factory_info.get("factory_error_rate"),
            # Total T states produced by the factory (PSSPC-transformed demand)
            "t_state_count": factory_info.get("t_state_count"),
        },
    )
