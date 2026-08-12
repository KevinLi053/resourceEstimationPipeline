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
# ---------------------------------------------------------------------------

# _TOL = 1e-12


# def _rz_to_bloqs(angle: float, eps: float = 1e-11) -> list:
#     """
#     Classify Rz(angle) for Qualtran synthesis purposes.

#     Returns ``[Rz(angle, eps)]`` so the rotation remains in the bloq graph and
#     can be synthesized by Qualtran's resource estimator (rather than being
#     pre-decomposed into T-gates).

#     Special angles that are exact Clifford operations (π/2, π, etc.) return
#     their exact gates to avoid unnecessary synthesis overhead.  Exact zeros
#     return an empty list (identity — no bloq needed).

#     Angle convention: Rz(θ) = exp(-iθ/2 · Z).

#     Parameters
#     ----------
#     angle : float  rotation angle in radians
#     eps   : float  synthesis precision for arbitrary rotations

#     Returns
#     -------
#     list of Qualtran bloq objects (may be empty for identity)
#     """
#     from qualtran.bloqs.basic_gates import SGate, TGate, ZGate, Rz

#     angle = float(angle) % (2 * math.pi)
#     if angle > math.pi:
#         angle -= 2 * math.pi

#     if abs(angle) < _TOL:
#         return []
#     if abs(angle - math.pi / 4) < _TOL:
#         return [TGate()]
#     if abs(angle + math.pi / 4) < _TOL:
#         return [TGate(is_adjoint=True)]
#     if abs(angle - math.pi / 2) < _TOL:
#         return [SGate()]
#     if abs(angle + math.pi / 2) < _TOL:
#         return [SGate(is_adjoint=True)]
#     if abs(abs(angle) - math.pi) < _TOL:
#         return [ZGate()]
#     if abs(angle - 3 * math.pi / 4) < _TOL:
#         return [SGate(), TGate()]
#     if abs(angle + 3 * math.pi / 4) < _TOL:
#         return [SGate(is_adjoint=True), TGate(is_adjoint=True)]
#     # All other angles → keep as an Rz bloq for Qualtran to synthesize.
#     return [Rz(angle, eps=eps)]


# ---------------------------------------------------------------------------
# Qiskit circuit → Qualtran CompositeBloq
# ---------------------------------------------------------------------------

def qiskit_to_composite_bloq(circuit: QuantumCircuit, eps: float = 1e-11):
    """
    Convert a Qiskit circuit to a Qualtran CompositeBloq.

    Rotation gates (Rz) are added as Qualtran bloqs so that
    Qualtran's resource estimator can synthesize them natively (e.g. into
    T-gates) during estimation.  This supports the workflow where
    ``rotation_synthesis_enabled=False`` is set in the transpilation config,
    allowing pre-synthesis to be skipped and letting Qualtran handle rotation
    synthesis with its own algorithms.

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
        CNOT, CZ, Hadamard, Rz, SGate, TGate,
        TwoBitSwap, XGate, YGate, ZGate,
    )

    _GATE_MAP = {
        "cx":    CNOT(),
        "cz":    CZ(),
        "h":     Hadamard(),
        "s":     SGate(),
        "sdg":   SGate(is_adjoint=True),
        "x":     XGate(),
        "y":     YGate(),
        "z":     ZGate(),
        "swap":  TwoBitSwap(),
        "t":     TGate(),
        "tdg":   TGate(is_adjoint=True),
    }

    _ROTATION_BLOQS = {"rz": Rz}
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
            qs[idx[0]] = bb.add(_ROTATION_BLOQS[name](angle, eps=eps), q=qs[idx[0]])

        elif name in _GATE_MAP and name not in ("cx", "cz", "swap"):
            qs[idx[0]] = bb.add(_GATE_MAP[name], q=qs[idx[0]])

        elif name == "cx":
            qs[idx[0]], qs[idx[1]] = bb.add(_GATE_MAP[name], ctrl=qs[idx[0]], target=qs[idx[1]])

        elif name == "cz":
            qs[idx[0]], qs[idx[1]] = bb.add(_GATE_MAP[name], q1=qs[idx[0]], q2=qs[idx[1]])

        elif name == "swap":
            qs[idx[0]], qs[idx[1]] = bb.add(_GATE_MAP[name], x=qs[idx[0]], y=qs[idx[1]])

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

# ---------------------------------------------------------------------------
# Data-block class (not instance) lookup helper
# ---------------------------------------------------------------------------

def _get_data_block_cls(db_type: str):
    """
    Return the DataBlock *class* (not an instance) for the given db_type string.

    Used by optimize_fifteen_to_one, which takes ``data_block_cls`` as a
    callable of the form ``data_block_cls(data_d=...)`` rather than a pre-built
    instance.  Mirrors the instance factory in ``_make_data_block``.
    (optimize_ccz2t delegates to Qualtran's grid search which always uses SimpleDataBlock.)
    """
    if db_type == "simple":
        from qualtran.surface_code import SimpleDataBlock
        return SimpleDataBlock
    if db_type == "compact":
        from qualtran.surface_code import CompactDataBlock
        return CompactDataBlock
    if db_type == "intermediate":
        from qualtran.surface_code import IntermediateDataBlock
        return IntermediateDataBlock
    # "fast" is the default; also serves as the fallback for unrecognised types.
    from qualtran.surface_code import FastDataBlock
    return FastDataBlock


import math
from qualtran.surface_code import (
    FifteenToOne,
    MultiFactory,
    LogicalErrorModel,
    FastDataBlock,
)
from qualtran.surface_code import beverland_et_al_model
from qualtran.resource_counting import GateCounts


def optimize_fifteen_to_one(
    *,
    n_logical_gates: GateCounts,
    logical_error_model: LogicalErrorModel,
    error_budget: float,
    algorithm,                      # AlgorithmSummary
    qec_scheme,                     # QECScheme, e.g. QECScheme.make_beverland_et_al()
    physical_error: float,
    rotation_model,                 # needed for Beverland c_min formula
    data_block_cls=FastDataBlock,   # swap for Compact/Intermediate/Simple as needed
    d_max: int = 25,
    fixed_point_iters: int = 10,
    cost_fn=lambda total_qubits, duration_cycles: total_qubits * duration_cycles,  # spacetime volume
    error_budget_fraction: float = 1 / 3,   # rotation share of error_budget
    return_pareto: bool = False,            # whether to return full Pareto front
):
    """
    Beverland/FifteenToOne optimizer: jointly search over FifteenToOne factory
    dimensions (d_X, d_Z, d_m) and parallel factory count for minimum cost.

    This function is specific to the Beverland et al. model + FifteenToOne
    factory.  It uses Beverland's minimum_time_steps() formula (Eq. D3) and
    beverland_et_al_model.code_distance() to size the data block.

    For the Gidney-Fowler/CCZ2T path use optimize_ccz2t() instead.

    Parameters
    ----------
    error_budget_fraction : float
        Fraction of ``error_budget`` reserved for rotation synthesis.
        The remaining fraction is shared between factory and data block errors
        by the optimizer through the total-error constraint:
        ``factory_err + data_block_err + rotation_err <= error_budget``.
    return_pareto : bool
        When True, returns a list of all Pareto-optimal solutions instead of
        the single cost-minimal one.  Each element is a dict with keys:
        ``factory, data_block, n_factories, time_steps, total_qubits,
        factory_error, data_block_error, rotation_error, total_error, cost``.
    """

    # ── Budget split ────────────────────────────────────────────────────
    rotation_budget = error_budget * error_budget_fraction
    remaining_budget = error_budget * (1 - error_budget_fraction)  # factory + data_block

    # The algorithm's own minimum runtime — the target the factories must keep pace with.
    c_min = beverland_et_al_model.minimum_time_steps(
        error_budget=remaining_budget, alg=algorithm, rotation_model=rotation_model
    )

    best = None  # (cost, factory, data_block, n_factories, time_steps, total_qubits,
                 #          factory_err, data_block_err, rotation_err, total_err, d_x, d_z, d_m)

    pareto: list[dict] = []   # only populated when return_pareto=True

    triples = [
        (3 + 6*k, 1 + 2*k, 1 + 2*k)
        for k in range(d_max)
    ]

    n_rotations = n_logical_gates.rotation
    rotation_err = n_rotations * (rotation_budget / max(n_rotations, 1)) if n_rotations > 0 else 0.0

    for d_x, d_z, d_m in triples:
    # --- Outer loop: the only real search. No closed form exists for factory error
    # (it comes from density-matrix simulation), so this has to be evaluated point by point. ---
        try:
            base = FifteenToOne(d_X=d_x, d_Z=d_z, d_m=d_m)
        except AssertionError:
            continue

        # Compute all three error components for this triple.
        factory_err = base.factory_error(n_logical_gates, logical_error_model)

        # Solve directly for the minimum n_factories to keep pace with c_min.
        single_cycles = base.n_cycles(n_logical_gates, logical_error_model)
        n_factories = max(1, math.ceil(single_cycles / c_min))
        factory = MultiFactory(base_factory=base, n_factories=n_factories)

        # --- Inner step: closed-form data_d resolution, NOT a sweep. ---
        # This is a fixed point because data_d depends on time_steps, and
        # (for non-Simple data blocks) the data block's own cycle count
        # depends on data_d. It converges in a couple of iterations since
        # data_d only moves in discrete odd steps.
        time_steps = factory.n_cycles(n_logical_gates, logical_error_model)
        data_block = None
        for _ in range(fixed_point_iters):
            data_d = beverland_et_al_model.code_distance(
                error_budget=remaining_budget,
                time_steps=time_steps,
                alg=algorithm,
                qec_scheme=qec_scheme,
                physical_error=physical_error,
            )
            data_block = data_block_cls(data_d=data_d)
            new_time_steps = max(
                factory.n_cycles(n_logical_gates, logical_error_model),
                data_block.n_cycles(n_logical_gates, logical_error_model),
            )
            if new_time_steps == time_steps:
                break
            time_steps = new_time_steps
        # --- end fixed point ---

        # Compute data block error using the converged distance.
        data_block_err = data_block.data_error(
            n_algo_qubits=algorithm.n_algo_qubits,
            n_cycles=time_steps,
            logical_error_model=logical_error_model,
        )

        total_err = factory_err + data_block_err + rotation_err
        if total_err > error_budget:
            continue

        total_qubits = factory.n_physical_qubits() + data_block.n_physical_qubits(
            n_algo_qubits=algorithm.n_algo_qubits
        )
        cost = cost_fn(total_qubits, time_steps)

        row = {
            "factory": factory,
            "data_block": data_block,
            "n_factories": n_factories,
            "time_steps": time_steps,
            "total_qubits": total_qubits,
            "factory_error": factory_err,
            "data_block_error": data_block_err,
            "rotation_error": rotation_err,
            "total_error": total_err,
            "cost": cost,
            "dx": d_x,
            "dz": d_z,
            "dm": d_m,
        }

        if return_pareto:
            pareto.append(row)

        if best is None or cost < best[0]:
            best = (cost, factory, data_block, n_factories, time_steps, total_qubits,
                    factory_err, data_block_err, rotation_err, total_err, d_x, d_z, d_m)

    if best is None:
        raise ValueError(
            f"No (d_X, d_Z, d_m) up to d_max={d_max} keeps total_error under "
            f"{error_budget:.3e} (factory {factory_err:.3e} + data_block 0 + rotation {rotation_err:.3e})."
        )

    cost, factory, data_block, n_factories, time_steps, total_qubits, \
        factory_err, data_block_err, rotation_err, total_err, d_x, d_z, d_m = best

    if return_pareto:
        pareto_front = []

        for candidate in pareto:
            dominated = False

            for other in pareto:
                if other is candidate:
                    continue

                if (
                    other["total_qubits"] <= candidate["total_qubits"]
                    and other["time_steps"] <= candidate["time_steps"]
                    and (
                        other["total_qubits"] < candidate["total_qubits"]
                        or other["time_steps"] < candidate["time_steps"]
                    )
                ):
                    dominated = True
                    break

            if not dominated:
                pareto_front.append(candidate)

        return pareto_front

    return {
        "factory": factory,
        "data_block": data_block,
        "n_factories": n_factories,
        "time_steps": time_steps,
        "total_qubits": total_qubits,
        "factory_error": factory_err,
        "data_block_error": data_block_err,
        "rotation_error": rotation_err,
        "total_error": total_err,
        "cost": cost,
        "dx": d_x,
        "dz": d_z,
        "dm": d_m,
    }

def optimize_ccz2t(
    *,
    n_logical_gates: GateCounts,
    logical_error_model: LogicalErrorModel,
    error_budget: float,
    algorithm,                          # AlgorithmSummary
    qec_scheme,                         # QECScheme — used for code_distance_from_budget
    physical_error: float,
    d1_min: int = 5,
    d1_max: int = 25,
    d2_max: int = 41,
    n_factories: int = 10,
    cost_fn=lambda total_qubits, duration_cycles: total_qubits * duration_cycles,
    error_budget_fraction: float = 1 / 3,   # rotation share of error_budget
    return_pareto: bool = False,
):
    """
    Gidney-Fowler/CCZ2T optimizer: iterates over factory configurations via
    ``iter_ccz2t_factories`` and for each derives the data block code distance
    analytically from the residual budget — the prescription from the
    Gidney-Fowler paper (arXiv:1812.01238).

    This avoids discretising data_d into a fixed grid (as ``get_ccz2t_costs_from_grid_search``
    does) and avoids the backwards-compatible ``get_ccz2t_costs`` wrapper it uses internally.
    For each factory configuration the optimal data_d satisfies:
        data_d = code_distance_from_budget(
            (remaining_budget − factory_err) / (n_tiles × n_cycles)
        )

    Differences from ``optimize_fifteen_to_one`` (Beverland/FifteenToOne):
      - Loops over (l1_d, l2_d) pairs; data_d is derived analytically per pair.
      - factory_error is from total_t_and_ccz_count() not total_t_count().
      - SimpleDataBlock (n_steps_to_consume_a_magic_state=0) → factory-cycle-limited.
      - Does NOT use beverland_et_al_model.minimum_time_steps() or code_distance().
      - Returns distances as l1_d/l2_d instead of dx/dz/dm.

    return_pareto=True returns a one-element list (single best solution).
    """
    from qualtran.surface_code import SimpleDataBlock
    from qualtran.surface_code.gidney_fowler_model import iter_ccz2t_factories, get_ccz2t_costs_from_grid_search
    from qualtran.surface_code.ccz2t_factory import CCZ2TFactory

    # ── Budget split ─────────────────────────────────────────────────────────
    rotation_budget = error_budget * error_budget_fraction
    remaining_budget = error_budget * (1 - error_budget_fraction)

    n_rotations = n_logical_gates.rotation
    rotation_err = (
        n_rotations * (rotation_budget / max(n_rotations, 1)) if n_rotations > 0 else 0.0
    )
    best = None
    for n_fac in range(1, n_factories+1):
        factories = iter_ccz2t_factories(
            n_factories=n_fac,
            l1_start=d1_min,
            l1_stop=d1_max,
            l2_stop=d2_max,
        )

        cost, factory, data_block = get_ccz2t_costs_from_grid_search(
            n_logical_gates=n_logical_gates,
            n_algo_qubits=algorithm.n_algo_qubits,
            phys_err=physical_error,
            error_budget=remaining_budget,
            factory_iter=factories,
            cost_function=lambda pc: pc.qubit_hours,
        )

        time_steps = factory.n_cycles(n_logical_gates, logical_error_model)
        total_qubits = factory.n_physical_qubits() + data_block.n_physical_qubits(n_algo_qubits=algorithm.n_algo_qubits)
        factory_err = factory.factory_error(n_logical_gates, logical_error_model)

        data_block_err = data_block.data_error(
            n_algo_qubits=algorithm.n_algo_qubits,
            n_cycles=time_steps,
            logical_error_model=logical_error_model,
        )

        total_err = rotation_err + data_block_err + factory_err
        if best is None or cost.qubit_hours < best["cost"]:
                base_factory = getattr(factory, "base_factory", factory)
                best = {
                    "factory":          factory,
                    "data_block":       data_block,
                    "n_factories":      n_fac,
                    "time_steps":       time_steps,
                    "total_qubits":     total_qubits,
                    "factory_error":    factory_err,
                    "data_block_error": data_block_err,
                    "rotation_error":   rotation_err,
                    "total_error":      total_err,
                    "cost":             cost.qubit_hours,
                    "l1_d":             base_factory.distillation_l1_d,
                    "l2_d":             base_factory.distillation_l2_d,
                }

    if best is None:
        raise ValueError(
            f"No (l1_d, l2_d) with l1 in [{d1_min},{d1_max}], l2 in [l1+2,{d2_max}] "
            f"keeps factory+data error under {remaining_budget:.3e}."
        )

    return [best] if return_pareto else best

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
                "n_factories":     cfg_d.n_factories,
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
    ts_per_rotation = BeverlandEtAlRotationCost.rotation_cost(eps_per_rotation).t
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
    # Allocate a configurable fraction of the budget for rotation synthesis;
    # the remainder is shared between factory and data block errors.
    rotation_fraction = qt_cfg.rotation_error_budget_fraction if hasattr(qt_cfg, 'rotation_error_budget_fraction') else (1 / 3)
    eps_rot_global = error_budget * rotation_fraction

    # Pass 1: temporary bloq with placeholder precision to count arbitrary rotations.
    # Placeholder precision is not used in final estimation — it only drives the
    # bloq graph so we can run from_bloq() and inspect the rotation count after
    # Qualtran's special-angle classification (_rz_to_bloqs).
    TEMP_EPS = 1e-6
    tmp_bloq = qiskit_to_composite_bloq(circuit, eps=TEMP_EPS)
    tmp_algo = AlgorithmSummary.from_bloq(tmp_bloq)
    rot_count = int(tmp_algo.n_logical_gates.rotation)

    # When the circuit was pre-synthesized (rotation_synthesis_enabled=True) and
    # no arbitrary rotations remain, set the effective fraction to 0 so the full
    # error budget is available to the factory+data_block optimizer.
    # rotation_fraction (from config) is preserved for display; effective_rotation_fraction
    # is what is passed to the factory optimizer and the sweep error decomposition.
    _pre_synthesized = (
        rot_count == 0 and config.transpile.rotation_synthesis_enabled
        and config.transpile.synthesis_strategy != "passthrough"
    )
    effective_rotation_fraction = 0.0 if _pre_synthesized else rotation_fraction
    if _pre_synthesized:
        print(
            "[qualtran] Pre-synthesized circuit detected (rot_count=0, synthesis=enabled). "
            "Setting rotation error_budget_fraction=0 so the full error budget is "
            "available for factory+data_block optimization."
        )

    eps_per_rotation = eps_rot_global / max(rot_count, 1)

    # Pass 2: rebuild the bloq with the correct per-rotation precision.
    bloq = qiskit_to_composite_bloq(circuit, eps=eps_per_rotation)

    # Logical gate summary (second pass — used for all reported values)
    algo = AlgorithmSummary.from_bloq(bloq)
    gc = algo.n_logical_gates

    # ── Determine estimation path ─────────────────────────────────────────────
    # optimize_factory=True: call the appropriate factory optimizer to jointly
    # search over factory parameters and parallel count for minimum space-time.
    #
    #   Gidney-Fowler / CCZ2T  → optimize_ccz2t()
    #   Beverland / FifteenToOne → optimize_fifteen_to_one()
    #
    # Falls back to the sweep when:
    #   • use_azure_parameters=True  (Azure-override mode must remain unchanged)
    #   • factory_type not recognised as CCZ2T or FifteenToOne
    _use_factory_opt = (
        qt_cfg.optimize_factory
        and not qt_cfg.use_azure_parameters
        and ((qt_cfg.use_gidney_fowler or qt_cfg.factory_type == "ccz2t")
        or (qt_cfg.use_beverland or qt_cfg.factory_type == "15to1"))
    )
    # Which sub-optimizer to use: CCZ2T (Gidney-Fowler) or FifteenToOne (Beverland).
    _use_ccz2t_opt = _use_factory_opt and (
        qt_cfg.use_gidney_fowler or (
            not qt_cfg.use_beverland and qt_cfg.factory_type in ("CCZ2T", "ccz2t")
        )
    )
    # These track what was actually used (optimization may differ from config).
    effective_n_factories = qt_cfg.n_factories
    effective_factory_type = qt_cfg.factory_type

    if _use_factory_opt:
        # ── Optimization path ─────────────────────────────────────────────────
        # Build QEC scheme, physical parameters, and logical error model from
        # config, then dispatch to the appropriate factory optimizer.
        # A PhysicalCostModel is constructed from the resulting (factory,
        # data_block) pair so that duration_hr and error are computed by the
        # same formulas as the sweep path.
        from qualtran.surface_code import (
            BeverlandEtAlRotationCost,
            QECScheme,
            PhysicalParameters,
            PhysicalCostModel as _PCM,
        )

        if qt_cfg.use_beverland:
            _qec_scheme = QECScheme.make_beverland_et_al()
            # Mirror what _make_cost_model does: use the preset PhysicalParameters
            # when phys_err matches the default, otherwise use the user's values.
            _phys_params = (
                PhysicalParameters.make_beverland_et_al()
                if qt_cfg.phys_err == 1e-3
                else PhysicalParameters(
                    physical_error=qt_cfg.phys_err,
                    cycle_time_us=qt_cfg.cycle_time_us,
                )
            )
        elif qt_cfg.use_gidney_fowler:
            _qec_scheme = QECScheme.make_gidney_fowler()
            _phys_params = (
                PhysicalParameters.make_gidney_fowler()
                if qt_cfg.phys_err == 1e-3
                else PhysicalParameters(
                    physical_error=qt_cfg.phys_err,
                    cycle_time_us=qt_cfg.cycle_time_us,
                )
            )
        else:
            # Custom path: build QEC scheme from config string.
            _qec_scheme = _make_qec_scheme(qt_cfg.qec_scheme)
            _phys_params = PhysicalParameters(
                physical_error=qt_cfg.phys_err, cycle_time_us=qt_cfg.cycle_time_us
            )

        _logical_error_model = LogicalErrorModel(
            physical_error=qt_cfg.phys_err, qec_scheme=_qec_scheme
        )

        # ── Dispatch to the model-specific optimizer ──────────────────────────
        if _use_ccz2t_opt:
            # Gidney-Fowler / CCZ2T path — uses l2_error() / total_t_and_ccz_count()
            # and sizes the data block from the budget remaining after factory error.
            opt = optimize_ccz2t(
                n_logical_gates=gc,
                logical_error_model=_logical_error_model,
                error_budget=error_budget,
                algorithm=algo,
                qec_scheme=_qec_scheme,
                physical_error=qt_cfg.phys_err,
                error_budget_fraction=effective_rotation_fraction,
                n_factories=qt_cfg.n_factories,
                return_pareto=True,
            )
        else:
            # Beverland / FifteenToOne path — uses beverland_et_al_model
            # minimum_time_steps() and code_distance() for data block sizing.
            opt = optimize_fifteen_to_one(
                n_logical_gates=gc,
                logical_error_model=_logical_error_model,
                error_budget=error_budget,
                algorithm=algo,
                qec_scheme=_qec_scheme,
                physical_error=qt_cfg.phys_err,
                rotation_model=BeverlandEtAlRotationCost,
                data_block_cls=_get_data_block_cls(qt_cfg.data_block),
                d_max=qt_cfg.optimize_factory_d_max,
                error_budget_fraction=effective_rotation_fraction,
                return_pareto=True,
            )

        # opt is now a Pareto front: list of dicts with component errors.
        if not opt:
            raise ValueError(
                f"Factory optimizer found no solutions within error_budget={error_budget:.3e}. "
                f"Try a larger budget or d_max."
            )

        # Build a PhysicalCostModel from the optimized components so that
        # duration_hr and total error (factory + data block) are computed
        # via the same internal formulas as the sweep path, and the existing
        # qubit-breakdown extraction code below works without modification.
        model = _PCM(
            physical_params=_phys_params,
            data_block=opt[0]["data_block"],
            factory=opt[0]["factory"],
            qec_scheme=_qec_scheme,
        )
        phys_qubits = opt[0]["total_qubits"]
        duration_hr = model.duration_hr(algo)
        error = model.error(algo)  # factory + data_block only (rotation excluded)
        data_d = opt[0]["data_block"].data_d
        effective_n_factories = opt[0]["n_factories"]
        effective_factory_type = "CCZ2TFactory" if _use_ccz2t_opt else "FifteenToOne"

        # Convert Pareto front into _pareto_rows with all component errors.
        _pareto_rows: list = []
        for row in opt:
            m = _PCM(
                physical_params=_phys_params,
                data_block=row["data_block"],
                factory=row["factory"],
                qec_scheme=_qec_scheme,
            )
            _pareto_rows.append({
                "data_d":           row["data_block"].data_d,
                "n_factories":      row["n_factories"],
                "physical_qubits":  row["total_qubits"],
                "duration_hr":      m.duration_hr(algo),
                "error":            m.error(algo),         # factory + data block only
                "factory_error":    row["factory_error"],
                "data_block_error": row["data_block_error"],
                "rotation_error":   row["rotation_error"],
                "total_error":      row["total_error"],
            })

    else:
        # ── Sweep path (existing behavior, preserved exactly) ─────────────────
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
        _pareto_rows = valid_rows

    # ── Add rotation error to sweep rows ────────────────────────────────
    if not _use_factory_opt:
        # For sweep path, each row already has factory_error + data_block_error in model.error().
        # Decompose and add rotation component.
        for r in _pareto_rows:
            if r["model"] is None or math.isnan(r["physical_qubits"]):
                continue
            m = r["model"]
            rot_err_row = rot_count * (eps_rot_global / max(rot_count, 1)) if rot_count > 0 else 0.0
            # Decompose model.error() into factory vs data_block.
            _logical_error_model_sweep = LogicalErrorModel(physical_error=qt_cfg.phys_err, qec_scheme=m.qec_scheme)
            try:
                f_err = m.factory.factory_error(algo, _logical_error_model_sweep)
            except Exception:
                f_err = 0.0
            r["factory_error"]   = f_err
            r["data_block_error"] = max(0.0, r["error"] - f_err)
            r["rotation_error"]  = rot_err_row
            r["total_error"]     = r["error"] + rot_err_row

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
                round(duration_hr * 3600 * 1e6 / cycle_time_us
            ) / data_d)

    # Circuit-derived T depth (DAG layer analysis on the transpiled Clifford+T circuit).
    # Qualtran's CompositeBloq has no time ordering so we derive T depth from the
    # transpiled input circuit directly instead. Same function used for Azure → identical values.
    circuit_t_depth = compute_t_depth(circuit)

    # Factory statistics (best-effort; attributes vary across Qualtran versions)
    factory_stats = _extract_factory_stats(model)

    # Factory description string (include total and per-factory for clarity).
    # Use effective_* values so the optimization path reports the actual
    # factory type and count chosen by the factory optimizer.
    factory_count_str = effective_factory_type
    if factory_qubits is not None:
        if effective_n_factories > 1:
            factory_count_str = (
                f"{effective_factory_type}×{effective_n_factories} ("
                # f"{factory_qubits:,} total qubits; "
                f"{qubits_per_factory:,} each)"
            )
        else:
            factory_count_str = f"{effective_factory_type}×1 ({factory_qubits:,} qubits)"

    if _use_ccz2t_opt:
        # CCZ2T factory distances: (l1_d, l2_d) instead of (dx, dz, dm).
        factory_ds = (opt[0]["l1_d"], opt[0]["l2_d"])
    elif _use_factory_opt:
        factory_ds = (opt[0]["dx"], opt[0]["dz"], opt[0]["dm"])
    else:
        factory_ds = None

    # ── Compute component errors for the selected solution ───────────────
    if _use_factory_opt:
        # Optimisation path: opt is a Pareto front; use the best (index 0).
        factory_err   = opt[0]["factory_error"]
        data_block_err = opt[0]["data_block_error"]
        rotation_err  = opt[0]["rotation_error"]
    else:
        # Sweep path: decompose model.error() (which is factory + data_block).
        _logical_error_model_sweep = LogicalErrorModel(physical_error=qt_cfg.phys_err, qec_scheme=model.qec_scheme)
        try:
            factory_err   = model.factory.factory_error(algo, _logical_error_model_sweep)
            data_block_err = model.data_block.data_error(algo.n_algo_qubits, int(model.n_cycles(algo)), _logical_error_model_sweep)
        except Exception:
            factory_err   = 0.0
            data_block_err = max(0.0, error - factory_err)
        rotation_err = rot_count * (eps_rot_global / max(rot_count, 1)) if rot_count > 0 else 0.0

    total_error = factory_err + data_block_err + rotation_err

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
        logical_error_rate=float(total_error),
        rotation_error=rotation_err,
        data_block_error=data_block_err,
        factory_error=factory_err,
        code_distance=data_d,
        factory_type=effective_factory_type,
        factory_tuple=factory_ds,
        factory_count=factory_count_str,
        num_factories=effective_n_factories,
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
            f"rotation_synthesis={'enabled' if config.transpile.rotation_synthesis_enabled else 'disabled'}; "
            f"synthesis_method={config.transpile.synthesis_method if config.transpile.rotation_synthesis_enabled else 'N/A (passthrough)'}; "
            f"basis={config.transpile.basis_gates}; "
            f"eps_per_rotation={eps_per_rotation:.2e} (derived from error_budget={error_budget:.0e}); "
            f"rot_count={rot_count}; "
            f"PauliEvolutionGate SuzukiTrotter order={config.evolution.synthesis_order} "
            f"reps={config.evolution.synthesis_reps}; "
            f"t={config.evolution.evolution_time}"
        ),
        architecture_assumptions=(
            f"{effective_factory_type} factory ×{effective_n_factories}; "
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
                {k: v for k, v in r.items() if k != "model"} for r in _pareto_rows
            ],
            # ── Rotation synthesis diagnostics ───────────────────────────────
            "rot_count_from_pass1": rot_count,
            "eps_per_rotation":     eps_per_rotation,
            "error_budget_global":  error_budget,
            "rotation_error_fraction": rotation_fraction,
            # ── Three-component error decomposition ────────────────────────
            "component_errors": {
                "factory":     factory_err,
                "data_block":  data_block_err,
                "rotation":    rotation_err,
                "total":       total_error,
            },
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
