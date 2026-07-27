"""
Qualtran Resource Estimator adapter.

Reuses implementations from:
  - qualtranEstimator/qualtranCircuitBuilder.ipynb  (Steps 5–12)
  - estimator/analysis/hamlib.ipynb                  (qiskit_to_composite_bloq)

The public entry point is :func:`estimate`, which satisfies the
:class:`~estimators.base.Estimator` protocol.

Notes on metric availability
-----------------------------
CompositeBloq schedules gates in a data-flow graph rather than a time-ordered
layer sequence, so T-depth and logical-depth (layer depth) are not directly
computable without explicit circuit scheduling — those fields remain None.

Logical cycles are derived from duration_hr and cycle_time_us:
    logical_cycles = duration_hr * 3_600 * 1_000_000 / cycle_time_us
This is exact when duration_hr is computed as (n_cycles * cycle_time_us / 1e6 / 3600).

error_budget is used to derive per-rotation synthesis precision:
    eps_per_rotation = (error_budget / 3) / rotation_count
The resulting logical_error_rate is computed from code distance by Qualtran.
"""
from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple, Dict

from qiskit import QuantumCircuit

from ..config import PipelineConfig, QualtranConfig
from ..circuit.transpile import compute_t_depth
from .base import EstimationResult


# ---------------------------------------------------------------------------
# Rz gate classification
# (source: qualtranCircuitBuilder.ipynb — cell-bloq-helpers / _rz_to_bloqs)
# (source: estimator/analysis/hamlib.ipynb — cell e6b886f8)
# ---------------------------------------------------------------------------

_TOL = 1e-12


def _rz_to_bloqs(angle: float, eps: float = 1e-11) -> list:
    """
    Classify Rz(angle) for Qualtran synthesis purposes.

    Returns ``[Rz(angle, eps)]`` so the rotation remains in the bloq graph and
    can be synthesized by Qualtran's resource estimator (rather than being
    pre-decomposed into T-gates).

    Special angles that are exact Clifford operations (π/2, π, etc.) return
    their exact gates to avoid unnecessary synthesis overhead.  Exact zeros
    return an empty list (identity — no bloq needed).

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
    # All other angles → keep as an Rz bloq for Qualtran to synthesize.
    return [Rz(angle, eps=eps)]


# ---------------------------------------------------------------------------
# Qiskit circuit → Qualtran CompositeBloq
# ---------------------------------------------------------------------------

def qiskit_to_composite_bloq(circuit: QuantumCircuit, eps: float = 1e-11):
    """
    Convert a Qiskit circuit to a Qualtran CompositeBloq.

    Rotation gates (Rz, Rx, Ry) are added as Qualtran bloqs so that
    Qualtran's resource estimator can synthesize them natively (e.g. into
    T-gates) during estimation.  This supports the workflow where
    ``rotation_synthesis_enabled=False`` is set in the transpilation config,
    allowing pre-synthesis to be skipped and letting Qualtran handle rotation
    synthesis with its own algorithms.

    Special Rz angles (multiples of π/4) are recognised as exact Clifford/T
    gates — no synthesis overhead is incurred for those.  Arbitrary angles
    become ``Rz(angle, eps)`` bloqs preserved for Qualtran's estimator.

    One named 1-qubit register per qubit (q0, q1, …) so the data-flow
    graph tracks each wire independently.

    Parameters
    ----------
    circuit : QuantumCircuit  (may contain arbitrary rotation gates Rz/Rx/Ry)
    eps     : float  synthesis precision for arbitrary rotation bloqs

    Returns
    -------
    qualtran.CompositeBloq
    """
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import (
        CNOT, CZ, Hadamard, MeasureZ, Rx, Rz, Ry, SGate, TGate, Toffoli,
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
        # Rotation gates — stored as None so they are instantiated per
        # instruction angle below.  Keeping them as Rz/Rx/Ry bloqs lets
        # Qualtran's resource estimator synthesize them natively (e.g. into
        # T-gates) during estimation rather than requiring pre-synthesis in
        # the transpilation stage.
        "rz":    None,
        "rx":    None,
        "ry":    None,
    }

    _ROTATION_BLOQS = {"rz": Rz, "rx": Rx, "ry": Ry}
    _IGNORED = {"barrier", "reset", "id", "measure"}

    n = circuit.num_qubits
    bb = BloqBuilder()
    qs = [bb.add_register(f"q{i}", 1) for i in range(n)]

    for instruction in circuit.data:
        name = instruction.operation.name
        if name in _IGNORED:
            continue

        idx = [circuit.find_bit(q).index for q in instruction.qubits]

        if name in _ROTATION_BLOQS:
            angle = instruction.operation.params[0]
            bloq_cls = _ROTATION_BLOQS[name]
            # For Rx/Ry: add the rotation bloq directly (no special-angle
            # classification needed — let Qualtran's estimator handle them).
            if name in ("rx", "ry"):
                qs[idx[0]] = bb.add(bloq_cls(angle, eps=eps), q=qs[idx[0]])
            else:
                # Rz: use _rz_to_bloqs to skip exact Clifford gates (e.g. T, S, Z)
                # at special angles, but keep arbitrary angles as Rz bloqs so
                # Qualtran can synthesize them during resource estimation.
                for rot_bloq in _rz_to_bloqs(angle, eps=eps):
                    if isinstance(rot_bloq, Rz):
                        qs[idx[0]] = bb.add(Rz(angle, eps=eps), q=qs[idx[0]])
                    else:
                        qs[idx[0]] = bb.add(rot_bloq, q=qs[idx[0]])

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
# Data block construction helper
# ---------------------------------------------------------------------------

def _make_data_block(db_type: str, data_d: int):
    """Return a Qualtran data block object for given type string"""
    
    if db_type == "simple":
        from qualtran.surface_code import SimpleDataBlock
        return SimpleDataBlock (data_d=data_d)
    if db_type == "compact":
        from qualtran.surface_code import CompactDataBlock
        return CompactDataBlock (data_d=data_d)
    if db_type == "intermediate":
        from qualtran.surface_code import IntermediateDataBlock
        return IntermediateDataBlock (data_d=data_d)
    if db_type == "fast":
        from qualtran.surface_code import FastDataBlock
        return FastDataBlock (data_d=data_d)

# ---------------------------------------------------------------------------
# Factory construction helper
# ---------------------------------------------------------------------------

def _make_qualtran_factory(factory_type: str, data_d: int = 17):
    """Return a Qualtran factory object for the given factory_type string."""
    if factory_type == "CCZ2T":
        from qualtran.surface_code import CCZ2TFactory
        return CCZ2TFactory()
    if factory_type == "15to1":
        from qualtran.surface_code import FifteenToOne
        return FifteenToOne(d_X=data_d, d_Z=data_d, d_m=data_d)
    raise ValueError(
        f"Unknown Qualtran factory type {factory_type!r}. Supported: 'CCZ2T', 'FifteenToOne'."
    )

# ---------------------------------------------------------------------------
# QEC scheme construction helper
# ---------------------------------------------------------------------------

def _make_qec_scheme(qec: str):
    """Return a QECScheme object for given qec_scheme string"""
    from qualtran.surface_code import QECScheme
    if qec == "beverland":
        return QECScheme.make_beverland_et_al()
    if qec == "gidney_fowler":
        return QECScheme.make_gidney_fowler()
    raise ValueError(
        f"Unknown Qualtran QEC scheme type {qec!r}. Supported: 'beverland', 'gidney_fowler'. "
    )

# ---------------------------------------------------------------------------
# PhysicalCostModel construction helpers
# (source: qualtranCircuitBuilder.ipynb — cell-phys-estimate / cell-hw-sweep)
# ---------------------------------------------------------------------------

def _wrap_factory(base_factory, n_factories: int):
    """
    Wrap a single factory in MultiFactory when n_factories > 1.

    MultiFactory multiplies the physical qubit footprint by n_factories and
    divides n_cycles by n_factories — matching Azure's parallel-factory model.
    n_factories=1 returns the factory unchanged.
    """
    if n_factories == 1:
        return base_factory
    from qualtran.surface_code import MultiFactory
    return MultiFactory(base_factory=base_factory, n_factories=n_factories)


def _make_cost_model(cfg: QualtranConfig):
    """
    Build a Qualtran PhysicalCostModel from the pipeline configuration.

    Source: qualtranCircuitBuilder.ipynb — cell-phys-estimate, cell-hw-sweep.
    When cfg.n_factories > 1, the base factory is wrapped in MultiFactory so
    that both qubit footprint and runtime scale correctly.
    """
    from qualtran.surface_code import PhysicalCostModel

    if cfg.use_gidney_fowler and cfg.phys_err == 1e-3:
        model = PhysicalCostModel.make_gidney_fowler(data_d=cfg.data_d)
        if cfg.n_factories > 1:
            from qualtran.surface_code import (
                PhysicalParameters
            )
            model = PhysicalCostModel(
                physical_params=model.physical_params,
                data_block=model.data_block,
                factory=_wrap_factory(model.factory, cfg.n_factories),
                qec_scheme=model.qec_scheme,
            )
        return model

    if cfg.use_beverland and cfg.phys_err == 1e-3:
        model = PhysicalCostModel.make_beverland_et_al(data_d=cfg.data_d, data_block_name="fast", factory_ds=(cfg.data_d, cfg.data_d, cfg.data_d))
        if cfg.n_factories > 1:
            model = PhysicalCostModel(
                physical_params=model.physical_params,
                data_block=model.data_block,
                factory=_wrap_factory(model.factory, cfg.n_factories),
                qec_scheme=model.qec_scheme,
            )
        return model

    # Custom hardware profile (varies phys_err, cycle_time_us, and factory_type)
    from qualtran.surface_code import PhysicalParameters
    base_factory = _make_qualtran_factory(cfg.factory_type, cfg.data_d)
    return PhysicalCostModel(
        physical_params=PhysicalParameters(
            physical_error=cfg.phys_err,
            cycle_time_us=cfg.cycle_time_us,
        ),
        data_block=_make_data_block(cfg.data_block, data_d=cfg.data_d),
        factory=_wrap_factory(base_factory, cfg.n_factories),
        qec_scheme=_make_qec_scheme(cfg.qec_scheme),
    )


def _sweep_distances(cfg: QualtranConfig, algo, *, azure_params: Optional[Dict[str, Optional[int]]] = None) -> list:
    """
    Sweep code distances and return list of (data_d, model, phys_qubits, duration_hr, error).

    Parameters
    ----------
    cfg : QualtranConfig
        Configuration for the sweep.
    algo : AlgorithmSummary
        Logical gate counts from ``AlgorithmSummary.from_bloq()``.
    azure_params : dict or None
        When provided and ``cfg.use_azure_parameters is True``, this dict
        overrides the distance sweep with Azure's chosen parameters:

          * ``"code_distance"`` → replace the full sweep with a single fixed
            distance (if not None).  Qualtran estimates resources at exactly
            that code distance.
          * ``"num_factories"`` → force ``n_factories`` to this value,
            regardless of what ``cfg.n_factories`` says.

        Values that are ``None`` leave the corresponding parameter unchanged.

    Returns
    -------
    list[dict]
        One row per distance (or one row total when Azure fixed-distance is used).
    """
    from qualtran.surface_code import PhysicalCostModel

    # ── Determine which distances to sweep ───────────────────────────────
    if azure_params is not None and cfg.use_azure_parameters:
        azure_d = azure_params.get("code_distance")
        if azure_d is not None:
            # Mode 1 (Azure-matched): sweep only Azure's chosen distance.
            distances = [azure_d]
        else:
            # Azure distance unavailable → fall back to native Qualtran behavior
            distances = cfg.data_d_sweep or [cfg.data_d]
    else:
        # Mode 2 (native Qualtran optimization): use config as-is
        distances = cfg.data_d_sweep or [cfg.data_d]

    # When Azure params are injected, num_factories may override cfg.n_factories
    az_factory = (
        azure_params.get("num_factories")
        if azure_params is not None and cfg.use_azure_parameters
        else None
    )

    rows = []  # Collect sweep results here
    for d in distances:
        try:
            cfg_d = QualtranConfig(
                data_d=d,
                n_factories=az_factory if az_factory is not None else cfg.n_factories,
                phys_err=cfg.phys_err,
                cycle_time_us=cfg.cycle_time_us,
                error_budget=cfg.error_budget,
                data_block=cfg.data_block,
                factory_type=cfg.factory_type,
                qec_scheme=cfg.qec_scheme,
                use_gidney_fowler=cfg.use_gidney_fowler,
                use_beverland=cfg.use_beverland,
            )
            m = _make_cost_model(cfg_d)
            rows.append({
                "data_d":          d,
                "n_factories":     cfg.n_factories,
                "model":           m,
                "physical_qubits": m.n_phys_qubits(algo),
                "duration_hr":     m.duration_hr(algo),
                "error":           m.error(algo),
            })
        except Exception as exc:
            rows.append({
                "data_d":          d,
                "n_factories":     cfg.n_factories,
                "model":           None,
                "physical_qubits": float("nan"),
                "duration_hr":     float("nan"),
                "error":           float("nan"),
                "_error_msg":      str(exc),
            })
    return rows


# ---------------------------------------------------------------------------
# Total T-count helper  (accounts for Rz synthesis)
# ---------------------------------------------------------------------------

def compute_total_t_count(algo, eps_per_rotation: float) -> Tuple[int, float]:
    """
    Compute total T-count given a pre-derived per-rotation synthesis precision.

    ``eps_per_rotation`` must already be derived from the global error budget as:
        eps_per_rotation = (error_budget / 3) / max(rotation_count, 1)

    Parameters
    ----------
    algo : AlgorithmSummary
        Logical gate counts from ``AlgorithmSummary.from_bloq()``.
    eps_per_rotation : float
        Synthesis precision allocated per arbitrary rotation.

    Returns
    -------
    (t_total, ts_per_rotation)
    """
    from qualtran.surface_code import BeverlandEtAlRotationCost
    ts_per_rotation = round(BeverlandEtAlRotationCost.rotation_cost(eps_per_rotation).t)
    t_total = int(algo.n_logical_gates.total_t_count(
        ts_per_toffoli=4, ts_per_cswap=4, ts_per_and_bloq=4,
        ts_per_rotation=ts_per_rotation,
    ))

    return t_total, float(ts_per_rotation)


# ---------------------------------------------------------------------------
# Factory statistics extraction
# ---------------------------------------------------------------------------

def _extract_factory_stats(model) -> dict:
    """
    Extract as many factory statistics as the factory object exposes.

    Returns a dict with keys present only when the attribute exists.
    All values are stored in `extra` for diagnostics.
    """
    stats: dict = {}
    factory = getattr(model, "factory", None)
    if factory is None:
        return stats

    stats["factory_class"] = type(factory).__name__

    # n_physical_qubits: already used for physical_factory_qubits
    for attr in ("n_physical_qubits", "n_phys_qubits", "num_physical_qubits",
                 "physical_qubits"):
        try:
            v = getattr(factory, attr, None)
            if v is not None:
                stats["n_physical_qubits"] = int(v)
                break
        except Exception:
            pass

    # Number of T states produced per distillation round (factory throughput)
    for attr in ("n_t_states_per_run", "n_magic_states_per_round", "n_states_per_run"):
        try:
            val = getattr(factory, attr)
            if val is not None:
                stats["t_states_per_round"] = int(val)
                break
        except Exception:
            pass

    # Output error rate per T state
    for attr in ("t_gate_error_rate", "distillation_error", "output_error"):
        try:
            val = getattr(factory, attr)
            if val is not None:
                stats["t_state_error_rate"] = float(val)
                break
        except Exception:
            pass

    # Number of distillation rounds / levels
    for attr in ("n_rounds", "n_distillation_rounds", "distillation_rounds"):
        try:
            val = getattr(factory, attr)
            if val is not None:
                stats["distillation_rounds"] = int(val)
                break
        except Exception:
            pass

    # Distillation time (in cycles)
    for attr in ("distillation_time_steps", "n_cycles"):
        try:
            val = getattr(factory, attr)
            if val is not None:
                stats["distillation_time_steps"] = int(val)
                break
        except Exception:
            pass

    return stats


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
      1. Derive ``eps_per_rotation`` from the global error budget (error_budget / 3).
      2. Pass 1: build a temporary CompositeBloq with a placeholder precision to
         count arbitrary rotations after Qualtran's special-angle classification.
      3. Compute the final ``eps_per_rotation = (error_budget / 3) / rotation_count``
         and rebuild the bloq for the second pass.
      4. Call ``AlgorithmSummary.from_bloq()`` for logical gate counts.
      5. Call ``PhysicalCostModel`` for physical resource estimates.
      6. If ``config.qualtran.data_d_sweep`` is set, sweep distances and pick
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
    error_budget = qt_cfg.error_budget if qt_cfg.error_budget is not None else 1e-3

    # ── Derive eps_per_rotation from the global error budget ──────────────
    # Reserve 1/3 of the global budget for rotation synthesis (same split as Azure).
    eps_rot_global = error_budget / 3

    # Pass 1: temporary bloq with placeholder precision to count arbitrary rotations.
    # Placeholder precision is not used in final estimation — it only drives the
    # bloq graph so we can run from_bloq() and inspect the rotation count after
    # Qualtran's special-angle classification (_rz_to_bloqs).
    TEMP_EPS = 1e-6
    tmp_bloq = qiskit_to_composite_bloq(circuit, eps=TEMP_EPS)
    tmp_algo = AlgorithmSummary.from_bloq(tmp_bloq)
    rot_count = int(tmp_algo.n_logical_gates.rotation)

    eps_per_rotation = eps_rot_global / max(rot_count, 1)

    # Pass 2: rebuild the bloq with the correct per-rotation precision.
    bloq = qiskit_to_composite_bloq(circuit, eps=eps_per_rotation)

    # Logical gate summary (second pass — used for all reported values)
    algo = AlgorithmSummary.from_bloq(bloq)
    gc = algo.n_logical_gates

    # Physical cost model (or sweep)
    rows = _sweep_distances(qt_cfg, algo)
    valid_rows = [r for r in rows if not math.isnan(r["physical_qubits"])]

    if not valid_rows:
        raise RuntimeError("Qualtran returned no valid Pareto solutions.")

    # Sort by physical qubits (ascending = default pareto_index=0)
    valid_rows.sort(key=lambda r: r["physical_qubits"] * r["duration_hr"])
    idx = qt_cfg.pareto_index % len(rows)
    best = valid_rows[idx]

    model = best["model"]
    phys_qubits = int(best["physical_qubits"])
    duration_hr = best["duration_hr"]
    error = best["error"]
    data_d = best["data_d"]

    # Phys qubit breakdown from the model (factory vs data block).
    # model.factory may be a MultiFactory (when qt_cfg.n_factories > 1), so
    # n_physical_qubits() already returns total_factory_qubits = n_factories × per_factory.
    # We also extract the per-factory footprint separately for debugging.
    factory_qubits: Optional[int] = None   # total factory qubits (all parallel factories)
    qubits_per_factory: Optional[int] = None  # single-factory footprint (diagnostic)
    compute_qubits: Optional[int] = None
    if model is not None:
        try:
            factory_qubits = int(model.factory.n_physical_qubits())
            compute_qubits = int(
                model.data_block.n_physical_qubits(n_algo_qubits=algo.n_algo_qubits)
            )
            # Resolve per-factory footprint: unwrap MultiFactory if needed.
            base_factory = getattr(model.factory, "base_factory", model.factory)
            qubits_per_factory = int(base_factory.n_physical_qubits())
        except Exception:
            # Fall back to subtraction if data_block API differs across versions.
            if factory_qubits is not None:
                compute_qubits = phys_qubits - factory_qubits

    # Exact T count from logical gate model (excludes Rz synthesis overhead).
    t_exact = int(gc.t) + 4 * int(gc.toffoli) + 4 * int(gc.and_bloq)

    # Total T count (including Rz synthesis) — uses the same eps_per_rotation.
    # When there are zero rotations, t_per_rotation is meaningless regardless of the
    # synthesis formula: it only describes "T gates per Rz" and there is no Rz to apply.
    # We also distinguish "genuinely no Rz" from "pre-synthesized into T".
    if rot_count > 0:
        t_total, t_per_rotation = compute_total_t_count(algo, eps_per_rotation=eps_per_rotation)
        synthesis_note: Optional[str] = None
    else:
        # No rotations to synthesize. Check transpile config to tell "pre-synthesized" from "genuinely no Rz".
        if config.transpile.rotation_synthesis_enabled:
            synthesis_note = "pre-synthesized (rotations converted to T)"
        else:
            synthesis_note = "no rotations"
        t_total = int(gc.t) + 4 * int(gc.toffoli) + 4 * int(gc.and_bloq)
        t_per_rotation = 0  # meaningful zero — no synthesis was performed

    # Logical cycles: derived from duration_hr and cycle_time_us.
    # Exact when duration_hr = n_cycles * cycle_time_us / 1e6 / 3600.
    logical_cycles: Optional[int] = None
    if model is not None and not math.isnan(duration_hr):
        cycle_time_us = model.physical_params.cycle_time_us
        if cycle_time_us > 0:
            logical_cycles = int(
                round(duration_hr * 3600 * 1e6 / cycle_time_us)
            ) / data_d

    # Circuit-derived T depth (DAG layer analysis on the transpiled Clifford+T circuit).
    # Qualtran's CompositeBloq has no time ordering so we derive T depth from the
    # transpiled input circuit directly instead. Same function used for Azure → identical values.
    circuit_t_depth = compute_t_depth(circuit)

    # Factory statistics (best-effort; attributes vary across Qualtran versions)
    factory_stats = _extract_factory_stats(model)

    # Factory description string (include total and per-factory for clarity)
    factory_count_str = qt_cfg.factory_type
    if factory_qubits is not None:
        if qt_cfg.n_factories > 1:
            factory_count_str = (
                f"{qt_cfg.factory_type}×{qt_cfg.n_factories} ("
                # f"{factory_qubits:,} total qubits; "
                f"{qubits_per_factory:,} each)"
            )
        else:
            factory_count_str = f"{qt_cfg.factory_type}×1 ({factory_qubits:,} qubits)"

    return EstimationResult(
        estimator_name=f"Qualtran (d={data_d}, p={model.physical_params.physical_error:.0e})",
        logical_qubits=algo.n_algo_qubits,
        # Logical depth requires explicit circuit scheduling;
        # CompositeBloq uses a data-flow graph (no time ordering) — not available.
        logical_depth=None,
        logical_cycles=logical_cycles,
        t_count=t_total,
        # Qualtran does not expose a separate "circuit" count since it works from bloqs.
        t_count_circuit=None,
        # T depth derived from the transpiled input circuit (DAG layer analysis).
        t_depth=circuit_t_depth,
        clifford_count=int(gc.clifford),
        rotation_count=int(gc.rotation),
        toffoli_count=int(gc.toffoli),
        measurement_count=int(gc.measurement),
        physical_qubits=phys_qubits,
        physical_compute_qubits=compute_qubits,
        # factory_qubits = n_factories × qubits_per_factory (MultiFactory handles this)
        physical_factory_qubits=factory_qubits,
        # Qualtran's SimpleDataBlock does not split memory from compute.
        physical_memory_qubits=None,
        runtime_seconds=duration_hr * 3600,
        # Set to global_error_budget when provided; otherwise None (Qualtran doesn't
        # natively accept a budget — it takes code distance instead).
        error_budget=qt_cfg.error_budget,
        logical_error_rate=float(error),
        code_distance=data_d,
        factory_type=qt_cfg.factory_type,
        factory_count=factory_count_str,
        num_factories=qt_cfg.n_factories,
        t_per_rotation=t_per_rotation,
        rotation_synthesis_precision=eps_per_rotation,
        synthesis_note=synthesis_note,
        physical_error_rate=model.physical_params.physical_error,
        cycle_time_us=model.physical_params.cycle_time_us,
        # Gate/measurement timing — informational (Qualtran PhysicalParameters
        # only stores cycle_time_us; these match the Beverland superconducting
        # defaults and are stored in QualtranConfig for display/comparison).
        gate_time_ns=qt_cfg.t_gate_ns if hasattr(qt_cfg, "t_gate_ns") else None,
        measurement_time_ns=qt_cfg.t_meas_ns if hasattr(qt_cfg, "t_meas_ns") else None,
        algorithm_assumptions=(
            f"Clifford+T circuit; basis={config.transpile.basis_gates}; "
            f"eps_per_rotation={eps_per_rotation:.2e} (derived from error_budget={error_budget:.0e}); "
            f"PauliEvolutionGate SuzukiTrotter order={config.evolution.synthesis_order} "
            f"reps={config.evolution.synthesis_reps}; "
            f"t={config.evolution.evolution_time}"
        ),
        architecture_assumptions=(
            f"{qt_cfg.factory_type} factory ×{qt_cfg.n_factories}; "
            f"surface-code d={data_d}; "
            f"phys_err={model.physical_params.physical_error:.0e}; "
            f"cycle_time={model.physical_params.cycle_time_us} µs; "
            f"t_gate={qt_cfg.t_gate_ns:.0f} ns; t_meas={qt_cfg.t_meas_ns:.0f} ns"
        ),
        raw=algo,
        extra={
            "algo_summary": {
                "t_exact":        t_exact,
                "t_total":        t_total,
                "rotation_count": int(gc.rotation),
                "clifford_count": int(gc.clifford),
                "toffoli_count":  int(gc.toffoli),
                "and_count":      int(gc.and_bloq),
                "measurement":    int(gc.measurement),
                "n_algo_qubits":  algo.n_algo_qubits,
            },
            "factory_stats": factory_stats,
            # per-factory footprint (before MultiFactory scaling) for debug output
            "qubits_per_factory": qubits_per_factory,
            "all_pareto_rows": [
                {k: v for k, v in r.items() if k != "model"} for r in valid_rows
            ],
            # ── Rotation synthesis diagnostics ───────────────────────────────
            "rot_count_from_pass1": rot_count,
            "eps_per_rotation":     eps_per_rotation,
            "error_budget_global":  error_budget,
            "placeholder_eps":      TEMP_EPS,
            # ── Cross-estimator timing alignment (Beverland formula) ─────────
            "qt_t_gate_ns": qt_cfg.t_gate_ns if hasattr(qt_cfg, "t_gate_ns") else None,
            "qt_t_meas_ns": qt_cfg.t_meas_ns if hasattr(qt_cfg, "t_meas_ns") else None,
            "qt_derived_cycle_time_ns": (
                int(qt_cfg.t_gate_ns) * 4 + int(qt_cfg.t_meas_ns) * 2
                if hasattr(qt_cfg, "t_gate_ns") and hasattr(qt_cfg, "t_meas_ns")
                else None
            ),
        },
    )
