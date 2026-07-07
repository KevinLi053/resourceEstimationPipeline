"""
Qualtran Resource Estimator adapter.

Reuses implementations from:
  - qualtranEstimator/qualtranCircuitBuilder.ipynb  (Steps 5–12)
  - estimator/analysis/hamlib.ipynb                  (qiskit_to_composite_bloq)

The public entry point is :func:`estimate`, which satisfies the
:class:`~resourceEstimationPipeline.estimators.base.Estimator` protocol.
"""
from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

from qiskit import QuantumCircuit

from resourceEstimationPipeline.config import PipelineConfig, QualtranConfig
from resourceEstimationPipeline.estimators.base import EstimationResult


# ---------------------------------------------------------------------------
# Rz gate classification
# (source: qualtranCircuitBuilder.ipynb — cell-bloq-helpers / _rz_to_bloqs)
# (source: estimator/analysis/hamlib.ipynb — cell e6b886f8)
# ---------------------------------------------------------------------------

_TOL = 1e-12


def _rz_to_bloqs(angle: float, eps: float = 1e-11) -> list:
    """
    Classify Rz(angle) into exact Qualtran basic gate(s) where possible.

    Source: qualtranCircuitBuilder.ipynb — cell-bloq-helpers.

    Special angles (multiples of π/4) map to exact Clifford/T gates —
    no synthesis overhead is incurred.  All other angles map to
    ``Rz(angle, eps)`` — one arbitrary rotation each.

    Angle convention: Rz(θ) = exp(-iθ/2 · Z).

    Parameters
    ----------
    angle : float  rotation angle in radians
    eps   : float  synthesis precision for arbitrary rotations

    Returns
    -------
    list of Qualtran bloq objects (may be empty for identity)
    """
    from qualtran.bloqs.basic_gates import SGate, TGate, ZGate, Rz

    angle = float(angle) % (2 * math.pi)
    if angle > math.pi:
        angle -= 2 * math.pi

    if abs(angle) < _TOL:
        return []
    if abs(angle - math.pi / 4) < _TOL:
        return [TGate()]
    if abs(angle + math.pi / 4) < _TOL:
        return [TGate(is_adjoint=True)]
    if abs(angle - math.pi / 2) < _TOL:
        return [SGate()]
    if abs(angle + math.pi / 2) < _TOL:
        return [SGate(is_adjoint=True)]
    if abs(abs(angle) - math.pi) < _TOL:
        return [ZGate()]
    if abs(angle - 3 * math.pi / 4) < _TOL:
        return [SGate(), TGate()]
    if abs(angle + 3 * math.pi / 4) < _TOL:
        return [SGate(is_adjoint=True), TGate(is_adjoint=True)]
    return [Rz(angle, eps=eps)]


# ---------------------------------------------------------------------------
# Qiskit circuit → Qualtran CompositeBloq
# (source: qualtranCircuitBuilder.ipynb — cell-bloq-helpers / qiskit_to_composite_bloq)
# (source: estimator/analysis/hamlib.ipynb — cell e6b886f8)
# ---------------------------------------------------------------------------

def qiskit_to_composite_bloq(circuit: QuantumCircuit, eps: float = 1e-11):
    """
    Convert a Qiskit circuit in the Clifford+T basis to a Qualtran CompositeBloq.

    Source: qualtranCircuitBuilder.ipynb — cell-bloq-helpers and cell-build-bloq.

    One named 1-qubit register per qubit (q0, q1, …) so the data-flow
    graph tracks each wire independently.

    Parameters
    ----------
    circuit : QuantumCircuit  (must be in CANONICAL_BASIS_GATES or QUALTRAN_EXTENDED_BASIS)
    eps     : float  synthesis precision for arbitrary Rz gates

    Returns
    -------
    qualtran.CompositeBloq
    """
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import (
        CNOT, CZ, Hadamard, MeasureZ, SGate, TGate, Toffoli,
        TwoBitSwap, XGate, YGate, ZGate,
    )

    _GATE_MAP = {
        "t":     TGate(),
        "tdg":   TGate(is_adjoint=True),
        "h":     Hadamard(),
        "x":     XGate(),
        "y":     YGate(),
        "z":     ZGate(),
        "s":     SGate(),
        "sdg":   SGate(is_adjoint=True),
        "cx":    CNOT(),
        "cz":    CZ(),
        "ccx":   Toffoli(),
        "swap":  TwoBitSwap(),
        "measure": MeasureZ(),
    }
    _IGNORED = {"barrier", "reset", "id", "measure"}

    n = circuit.num_qubits
    bb = BloqBuilder()
    qs = [bb.add_register(f"q{i}", 1) for i in range(n)]

    for instruction in circuit.data:
        name = instruction.operation.name
        if name in _IGNORED:
            continue

        idx = [circuit.find_bit(q).index for q in instruction.qubits]

        if name == "rz":
            angle = instruction.operation.params[0]
            for bloq in _rz_to_bloqs(angle, eps=eps):
                qs[idx[0]] = bb.add(bloq, q=qs[idx[0]])

        elif name in _GATE_MAP and name not in ("cx", "cz", "swap", "ccx"):
            qs[idx[0]] = bb.add(_GATE_MAP[name], q=qs[idx[0]])

        elif name == "cx":
            qs[idx[0]], qs[idx[1]] = bb.add(CNOT(), ctrl=qs[idx[0]], target=qs[idx[1]])

        elif name == "cz":
            qs[idx[0]], qs[idx[1]] = bb.add(CZ(), q1=qs[idx[0]], q2=qs[idx[1]])

        elif name == "swap":
            qs[idx[0]], qs[idx[1]] = bb.add(TwoBitSwap(), x=qs[idx[0]], y=qs[idx[1]])

        elif name == "ccx":
            ctrl_out, qs[idx[2]] = bb.add(
                Toffoli(), ctrl=(qs[idx[0]], qs[idx[1]]), target=qs[idx[2]]
            )
            qs[idx[0]], qs[idx[1]] = ctrl_out

    return bb.finalize(**{f"q{i}": qs[i] for i in range(n)})


# ---------------------------------------------------------------------------
# Factory construction helper
# ---------------------------------------------------------------------------

def _make_qualtran_factory(factory_type: str):
    """Return a Qualtran factory object for the given factory_type string."""
    if factory_type == "CCZ2T":
        from qualtran.surface_code import CCZ2TFactory
        return CCZ2TFactory()
    raise ValueError(
        f"Unknown Qualtran factory type {factory_type!r}. Supported: 'CCZ2T'."
    )


# ---------------------------------------------------------------------------
# PhysicalCostModel construction helpers
# (source: qualtranCircuitBuilder.ipynb — cell-phys-estimate / cell-hw-sweep)
# ---------------------------------------------------------------------------

def _make_cost_model(cfg: QualtranConfig):
    """
    Build a Qualtran PhysicalCostModel from the pipeline configuration.

    Source: qualtranCircuitBuilder.ipynb — cell-phys-estimate, cell-hw-sweep.
    """
    from qualtran.surface_code import PhysicalCostModel

    if cfg.use_gidney_fowler and cfg.phys_err == 1e-3:
        return PhysicalCostModel.make_gidney_fowler(data_d=cfg.data_d)

    if cfg.use_beverland and cfg.phys_err == 1e-3:
        return PhysicalCostModel.make_beverland_et_al(data_d=cfg.data_d)
    
    # Custom hardware profile (varies phys_err, cycle_time_us, and factory_type)
    from qualtran.surface_code import (
        PhysicalParameters, QECScheme, SimpleDataBlock,
    )
    return PhysicalCostModel(
        physical_params=PhysicalParameters(
            physical_error=cfg.phys_err,
            cycle_time_us=cfg.cycle_time_us,
        ),
        data_block=SimpleDataBlock(data_d=cfg.data_d),
        factory=_make_qualtran_factory(cfg.factory_type),
        qec_scheme=QECScheme.make_gidney_fowler(),
    )


def _sweep_distances(cfg: QualtranConfig, algo) -> list:
    """
    Sweep code distances and return list of (data_d, model, phys_qubits, duration_hr, error).

    Source: qualtranCircuitBuilder.ipynb — cell-sweep-d.
    """
    from qualtran.surface_code import PhysicalCostModel

    distances = cfg.data_d_sweep or [cfg.data_d]
    rows = []
    for d in distances:
        try:
            cfg_d = QualtranConfig(
                data_d=d,
                phys_err=cfg.phys_err,
                cycle_time_us=cfg.cycle_time_us,
                rz_eps=cfg.rz_eps,
                factory_type=cfg.factory_type,
                use_gidney_fowler=cfg.use_gidney_fowler,
                use_beverland=cfg.use_beverland,
            )
            m = _make_cost_model(cfg_d)
            rows.append({
                "data_d":          d,
                "model":           m,
                "physical_qubits": m.n_phys_qubits(algo),
                "duration_hr":     m.duration_hr(algo),
                "error":           m.error(algo),
            })
        except Exception as exc:
            rows.append({
                "data_d":          d,
                "model":           None,
                "physical_qubits": float("nan"),
                "duration_hr":     float("nan"),
                "error":           float("nan"),
                "_error_msg":      str(exc),
            })
    return rows


# ---------------------------------------------------------------------------
# Total T-count helper  (accounts for Rz synthesis)
# (source: estimator/analysis/hamlib.ipynb — cell ff536b72 / qualtran_compute_total_t_count)
# ---------------------------------------------------------------------------

def compute_total_t_count(algo, error_budget: float = 1e-3) -> int:
    """
    Estimate total T count including gates synthesised from arbitrary Rz rotations.

    Solovay-Kitaev synthesis requires O(log(1/ε)) T gates per rotation.
    This function uses the standard approximation: ~3 * log2(1/ε) T gates.

    Source: estimator/analysis/hamlib.ipynb.

    Parameters
    ----------
    algo         : AlgorithmSummary
    error_budget : float  target synthesis precision per rotation

    Returns
    -------
    int  estimated total T count
    """
    gc = algo.n_logical_gates
    t_exact = int(gc.t) + 4 * int(gc.toffoli) + 4 * int(gc.and_bloq)
    t_per_rz = max(1, int(3 * math.log2(1.0 / max(error_budget, 1e-15))))
    t_from_rz = int(gc.rotation) * t_per_rz
    return t_exact + t_from_rz


# ---------------------------------------------------------------------------
# Public estimator function
# ---------------------------------------------------------------------------

def estimate(
    circuit: QuantumCircuit,
    config: PipelineConfig,
) -> EstimationResult:
    """
    Run Qualtran's surface-code resource estimation on a Clifford+T circuit.

    Source: qualtranCircuitBuilder.ipynb — Steps 6–12.

    Pipeline:
      1. Convert the Qiskit circuit to a Qualtran ``CompositeBloq``.
      2. Call ``AlgorithmSummary.from_bloq()`` for logical gate counts.
      3. Call ``PhysicalCostModel`` for physical resource estimates.
      4. If ``config.qualtran.data_d_sweep`` is set, sweep distances and pick
         the best solution at ``config.qualtran.pareto_index``.

    Parameters
    ----------
    circuit : QuantumCircuit
        Canonical Clifford+T circuit (same one sent to the Azure estimator).
    config  : PipelineConfig

    Returns
    -------
    EstimationResult
    """
    from qualtran.surface_code import AlgorithmSummary

    qt_cfg = config.qualtran

    # 1. Qiskit → CompositeBloq
    bloq = qiskit_to_composite_bloq(circuit, eps=qt_cfg.rz_eps)

    # 2. Logical gate summary
    algo = AlgorithmSummary.from_bloq(bloq)
    gc = algo.n_logical_gates

    # 3. Physical cost model (or sweep)
    rows = _sweep_distances(qt_cfg, algo)
    valid_rows = [r for r in rows if not math.isnan(r["physical_qubits"])]

    if not valid_rows:
        raise RuntimeError("Qualtran returned no valid Pareto solutions.")

    # Sort by physical qubits (ascending = default pareto_index=0)
    valid_rows.sort(key=lambda r: r["physical_qubits"])
    idx = qt_cfg.pareto_index % len(valid_rows)
    best = valid_rows[idx]

    model = best["model"]
    phys_qubits = int(best["physical_qubits"])
    duration_hr = best["duration_hr"]
    error = best["error"]
    data_d = best["data_d"]

    # Phys qubit breakdown from the model (factory vs data block)
    factory_qubits: Optional[int] = None
    compute_qubits: Optional[int] = None
    try:
        factory_qubits = int(model.factory.n_physical_qubits)
        compute_qubits = phys_qubits - factory_qubits
    except Exception:
        pass

    # T count (including Rz synthesis)
    t_exact = int(gc.t) + 4 * int(gc.toffoli) + 4 * int(gc.and_bloq)
    t_total = compute_total_t_count(algo, error_budget=qt_cfg.rz_eps)
    t_per_rz_estimate = (
        (t_total - t_exact) // max(1, int(gc.rotation))
        if int(gc.rotation) > 0 else None
    )

    return EstimationResult(
        estimator_name=f"Qualtran (d={data_d}, p={qt_cfg.phys_err:.0e})",
        logical_qubits=algo.n_algo_qubits,
        t_count=t_total,
        t_depth=None,          # CompositeBloq does not compute T-depth directly
        clifford_count=int(gc.clifford),
        rotation_count=int(gc.rotation),
        toffoli_count=int(gc.toffoli),
        measurement_count=int(gc.measurement),
        physical_qubits=phys_qubits,
        physical_compute_qubits=compute_qubits,
        physical_factory_qubits=factory_qubits,
        physical_memory_qubits=None,
        runtime_seconds=duration_hr * 3600,
        error_budget=None,     # Qualtran takes code distance, not error budget directly
        logical_error_rate=float(error),
        code_distance=data_d,
        factory_count=qt_cfg.factory_type,
        t_per_rotation=t_per_rz_estimate,
        algorithm_assumptions=(
            f"Clifford+T circuit; basis={config.transpile.basis_gates}; "
            f"rz_eps={qt_cfg.rz_eps:.0e}; "
            f"PauliEvolutionGate SuzukiTrotter order={config.evolution.synthesis_order} "
            f"reps={config.evolution.synthesis_reps}; "
            f"t={config.evolution.evolution_time}"
        ),
        architecture_assumptions=(
            f"{qt_cfg.factory_type} factory; surface-code d={data_d}; "
            f"phys_err={qt_cfg.phys_err:.0e}; "
            f"cycle_time={qt_cfg.cycle_time_us} µs"
        ),
        raw=algo,
        extra={
            "algo_summary": {
                "t_exact":       t_exact,
                "t_total":       t_total,
                "rotation_count": int(gc.rotation),
                "clifford_count": int(gc.clifford),
                "toffoli_count":  int(gc.toffoli),
                "and_count":      int(gc.and_bloq),
                "measurement":    int(gc.measurement),
                "n_algo_qubits":  algo.n_algo_qubits,
            },
            "all_pareto_rows": [
                {k: v for k, v in r.items() if k != "model"} for r in valid_rows
            ],
        },
    )
