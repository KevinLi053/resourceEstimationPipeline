"""
Central configuration for the resource-estimation pipeline.

Changing parameters for either estimator requires editing only this file.
All tunable knobs are grouped by concern: Hamiltonian loading, circuit
construction, transpilation, Azure QDK, and Qualtran.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


# ---------------------------------------------------------------------------
# Hamiltonian loading
# ---------------------------------------------------------------------------

@dataclass
class HamlibConfig:
    """Controls which Hamiltonian is loaded from a HamLib HDF5 file."""

    hdf5_path: str = "./../hamlib/condensedmatter/heisenberg/heis.hdf5"
    """Path to the HamLib HDF5 file (relative to the repo root or absolute)."""

    key: Optional[str] = None
    """Specific dataset key. If None, `key_index` is used."""

    key_index: int = 313
    """Index into the list of all keys in the file (used when `key` is None)."""


# ---------------------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------------------

@dataclass
class EvolutionConfig:
    """Controls the PauliEvolutionGate circuit construction."""

    evolution_time: float = 1.0
    """Total Hamiltonian evolution time t."""

    synthesis_order: int = 2
    """SuzukiTrotter order (1 = 1st-order, 2 = 2nd-order, etc.)."""

    synthesis_reps: int = 10
    """Number of synthesis repetitions for SuzukiTrotter."""


# ---------------------------------------------------------------------------
# Transpilation
# ---------------------------------------------------------------------------

# Intermediate basis used during stages 1–2 of transpilation.
# Keeps arbitrary rotation gates so they can be optimised (combined/cancelled)
# before rotation synthesis runs.
INTERMEDIATE_BASIS_GATES: List[str] = [
    "cx", "rz", "rx", "ry", "h", "s", "sdg", "t", "tdg", "x", "y", "z",
]

# Pure Clifford+T basis — no arbitrary rotation gates.
# This is the output basis after rotation synthesis and is what both estimators
# receive.  Azure QDK and Qualtran are both able to process this gate set.
PURE_CLIFFORD_T_BASIS_GATES: List[str] = [
    "h", "s", "sdg", "t", "tdg", "cx", "x", "y", "z",
]

# Canonical Clifford+T basis accepted by both Azure QDK and Qualtran.
# Kept for backward compatibility; includes rz so that passthrough mode
# (rotation_synthesis_enabled=False) still works.
CANONICAL_BASIS_GATES: List[str] = [
    "cx", "rz", "h", "s", "sdg", "x", "y", "z", "t", "tdg",
]

# Qualtran can additionally accept these gates natively (superset).
QUALTRAN_EXTENDED_BASIS_GATES: List[str] = CANONICAL_BASIS_GATES + [
    "cz", "ccx", "swap",
]


@dataclass
class TranspileConfig:
    """Controls transpilation to the Clifford+T basis."""

    basis_gates: List[str] = field(
        default_factory=lambda: list(CANONICAL_BASIS_GATES)
    )
    """
    Gate set used in passthrough mode (rotation_synthesis_enabled=False).
    When synthesis is enabled this field is ignored — the pipeline uses
    INTERMEDIATE_BASIS_GATES for stages 1–2 and PURE_CLIFFORD_T_BASIS_GATES
    for the final stage automatically.
    """

    optimization_level: int = 1
    """
    Qiskit transpiler optimization level applied during stage 2 (while
    rotation gates still exist so they can be combined before synthesis).
    0 = decompose only.  1 = light (default).  2–3 = heavier.
    """

    seed_transpiler: Optional[int] = 42
    """Random seed for the transpiler (determinism)."""

    rotation_synthesis_enabled: bool = True
    """
    When True (default), arbitrary rotation gates (Rz/Rx/Ry) are synthesised
    into Clifford+T before estimation so both estimators receive an identical
    pure Clifford+T circuit.

    When False, the pipeline falls back to a single-stage transpile that
    passes rotation gates through verbatim according to ``basis_gates``.
    """

    rotation_synthesis_epsilon: float = 1e-11
    """
    Target approximation precision for Rz/Rx/Ry → Clifford+T synthesis.
    A smaller value increases accuracy at the cost of more T gates per rotation.
    Typical range: 1e-8 (fast, loose) to 1e-12 (slow, tight).
    """

    synthesis_strategy: str = "qiskit_synth"
    """
    Legacy control kept for backward compatibility.
    Prefer ``rotation_synthesis_enabled`` for new code.

    ``"qiskit_synth"`` → rotation_synthesis_enabled=True
    ``"passthrough"``  → rotation_synthesis_enabled=False
    """

    synthesis_method: str = "solovay_kitaev"
    """
    Which algorithm to use for synthesising arbitrary rotation gates into Clifford+T.

    ``"solovay_kitaev"`` (default) — uses Qiskit's built-in Solovay-Kitaev
        decomposition or its transpile-based fallback.  Works for all angles but
        produces approximate decompositions with T-count scaling ~log^3(1/epsilon).

    ``"pygridsynth"`` — uses the pygridsynth library for optimal or near-optimal
        exact Clifford+T synthesis via grid-based lattice search.  For angles that
        lie on the ``Z[1/2]`` Clifford+T grid (e.g. ``pi/4``, ``pi/8``, ``pi/6``)
        it returns the **minimal** T-count.  Typically produces 2-10x fewer T gates
        than Solovay-Kitaev for common angles.

    Requires ``synthesis_method="pygridsynth"`` to also have ``rotation_synthesis_enabled=True``.
    When synthesis is disabled (passthrough mode), this field is ignored.
    """

    pygridsynth_precision: Optional[float] = None
    """
    Dedicated approximation precision for the pygridsynth backend.

    If ``None`` (default), falls back to ``rotation_synthesis_epsilon``.
    Recommended range: 1e-8 (fast, near-optimal) to 1e-10 (optimal for exact grid points).

    Only relevant when ``synthesis_method="pygridsynth"``.
    """


# ---------------------------------------------------------------------------
# Azure QDK Resource Estimator
# ---------------------------------------------------------------------------

@dataclass
class AzureConfig:
    """
    Knobs for the Microsoft QDK Resource Estimator (`qdk.qre`).

    Reference implementations in: estimator/circuitBuilderGeneralized.ipynb
    """

    # Error budget
    error_budget: float = 0.01
    """Total tolerable algorithmic error (e.g. 0.01 = 1%)."""

    # Physical hardware model (GateBased)
    error_rate: float = 1e-3
    """Physical gate error rate."""

    gate_time_ns: float = 50.0
    """Single-qubit gate time in nanoseconds."""

    measurement_time_ns: float = 100.0
    """Measurement time in nanoseconds."""

    two_qubit_gate_time_ns: Optional[float] = None
    """Two-qubit gate time in nanoseconds."""

    # QEC: surface-code distance
    code_distance: Optional[int] = None
    """
    Fix the surface-code distance passed to SurfaceCode.q(distance=...).
    """

    # Magic-state factory
    factory_type: str = "RoundBased"
    """
    Factory model: 'Litinski19' or 'RoundBased'.
    Litinski19Factory is newer and generally more efficient.
    """

    slow_down_factors: List[float] = field(default_factory=lambda: [1.0])
    """
    Factory slow-down factors to sweep.
    Higher value → factory is slower but uses fewer qubits.
    """

    # Optimisation
    optimization_level: int = 1
    """
    Optimisation level forwarded to the estimator query.
    Higher = more aggressive gate cancellation before estimation.
    """

    # Use-graph
    use_graph: bool = True
    """
    use_graph=False for more completeness on Pareto frontier
    """

    use_qualtran_parameters: bool = False
    """
    When True, the Azure estimator will override its cycle time with
    values extracted from Qualtran's PhysicalParams.  This enables
    ``Mode 1: Qualtran-matched estimation`` — Azure uses Qualtrans's chosen
    cycle time as fixed override input instead of its own sweep or defaults.

    When False (default), Azure behaves normally with its own native
    parameter selection, enabling ``Mode 2: Native Azure optimization``.
    """

    minimize: str = "qubit_hours"

    # Estimation result selection
    pareto_index: int = 0
    """
    Which Pareto-optimal solution to use as the 'best' result.
    0 = minimum physical qubits (default), -1 = minimum runtime.
    """


# ---------------------------------------------------------------------------
# Qualtran Resource Estimator
# ---------------------------------------------------------------------------

@dataclass
class QualtranConfig:
    """
    Knobs for Google Qualtran's surface-code resource estimation pipeline.
    """

    # QEC: code distance
    data_d: int = 17
    """
    Surface-code data block distance.
    Higher = fewer logical errors, more physical qubits.
    """

    data_d_sweep: Optional[List[int]] = None
    """
    If set, override `data_d` and sweep all distances in this list.
    Example: list(range(7, 30, 2))
    """

    # Physical hardware parameters
    phys_err: float = 1e-3
    """Physical gate error rate (same default as Azure for apples-to-apples comparison)."""

    t_gate_ns: float = 50.0
    """Single-qubit gate time in nanoseconds — matches Beverland superconducting default."""

    t_meas_ns: float = 100.0
    """Measurement time in nanoseconds — matches Beverland superconducting default.

    Note: qualtran's PhysicalParameters only stores cycle_time_us, not these individually.
    They are retained here for display / comparison and to derive the Beverland formula
    (cycle_time_ns = 4*t_gate + 2*t_meas) when needed.
    """

    cycle_time_us: float = 1.0
    """Surface-code cycle time in microseconds."""

    # Rotation synthesis (error-budget driven)
    # eps_per_rotation is derived from error_budget at runtime as:
    #   eps_per_rotation = (error_budget / 3) / max(rotation_count, 1)
    # No separate rz_eps knob — this avoids conflicting error models.
    error_budget: float = 0.01

    # Data block type for logical qubit encoding
    data_block: str = "simple"
    """Data block type: 'simple', 'compact', 'intermediate', or 'fast'."""

    # Magic-state factory
    factory_type: str = "15to1"
    """
    Qualtran magic-state factory model used in the custom cost-model path.
    Currently supported: 'CCZ2T' (Gidney-Fowler CCZ-to-T factory) and 'FifteenToOne.
    Only takes effect when use_gidney_fowler=False and use_beverland=False.
    """
    
    # Quantum error correction scheme
    qec_scheme: str = "beverland"
    """
    QEC scheme for the custom cost model path.
    Currently supported: 'beverland', 'gidney_fowler'.
    Only takes effect when use_gidney_fowler=False and use_beverland=False.
    """

    # Number of parallel magic-state factories
    n_factories: int = 1
    """
    Number of parallel CCZ2T factories to run simultaneously.
    1 = single factory (default): runtime scales with T count, qubit footprint is fixed.
    N > 1: wraps with MultiFactory — footprint × N, runtime / N.
    Use this to match Azure's multi-factory behaviour.
    """

    # Decomposition / estimation precision
    use_gidney_fowler: bool = False
    """
    If True, use the Gidney-Fowler CCZ2T factory model.
    If False, construct a custom PhysicalCostModel from `phys_err` and `cycle_time_us`.
    """

    use_beverland: bool = False
    """
    If True, use the Beverland FifteenToOne factory model.
    If False, construct a custom PhysicalCostModel from `phys_err` and `cycle_time_us`.
    """

    # Azure parameter injection (connector)
    use_azure_parameters: bool = False
    """
    When True, the Qualtran estimator will override its QEC parameters with
    values extracted from an Azure QDK estimation result.  This enables
    ``Mode 1: Azure-matched estimation`` — Qualtran uses Azure's chosen
    code distance and factory count as fixed inputs instead of its own sweep
    or defaults.

    When False (default), Qualtran behaves normally with its own native
    parameter selection, enabling ``Mode 2: Native Qualtran optimization``.

    To activate in the pipeline, set this to True AND pass an
    ``azure_result`` via ``PipelineConfig.qualtran.azure_result``.
    If the Azure result is missing or does not expose code distance / factory
    count, the estimator falls back to its native behavior for each missing
    parameter.
    """

    # Azure QDK alignment
    azure_cycle_time_us: Optional[float] = None
    """
    Physical cycle time to use for the Azure estimator when it shares hardware
    parameters with qualtran (same SurfaceCode code_cycle_override).

    When ``None`` (default), this is derived from whichever qualtran preset is active:

      * ``use_beverland=True``        → 0.4 µs  (Beverland superconducting: 4×50 + 2×100)
      * ``use_gidney_fowler=True``    → 1.0 µs  (Gidney-Fowler default)
      * custom path                   → ``cycle_time_us``

    Set this explicitly to override the automatic derivation with a known value.
    """

    # Native factory optimization
    optimize_factory: bool = False
    """
    When True (and use_azure_parameters=False, use_gidney_fowler=False), call
    optimize_factory_and_count() instead of the fixed-distance sweep to jointly
    optimize the FifteenToOne factory dimensions (d_X, d_Z, d_m) and the number
    of parallel factories for minimum space-time volume.

    Applies to:
      - Qualtran native with use_beverland=True
      - Qualtran native custom path when factory_type="15to1"

    Silently skipped (falls back to sweep) when:
      - use_azure_parameters=True  (Azure-override mode is unchanged)
      - use_gidney_fowler=True     (CCZ2T factory, not FifteenToOne)
      - factory_type != "15to1"    (CCZ2T or other factory)

    Has no effect on Azure estimation; Azure QDK handles its own optimization.
    """

    optimize_factory_d_max: int = 15
    """
    Maximum FifteenToOne code distance to search during factory optimization.

    optimize_factory_and_count() sweeps all (d_X, d_Z, d_m) up to this bound.
    The search is O(d_max^3) and each combination calls FifteenToOne.factory_error()
    which is computationally expensive (~0.09 s per point).
    Practical guidance:
      - d_max=9  : ~10 s — suitable for quick experiments
      - d_max=15 : ~55 s — default, good balance of speed and coverage
      - d_max=25 : ~4 min — full search, needed only for very high accuracy

    Only used when optimize_factory=True.
    """

    # Result selection
    pareto_index: int = 0
    """
    Which Pareto-optimal solution (from the data_d sweep) to use as the 'best'.
    0 = minimum physical qubits, -1 = minimum runtime.
    """

# ---------------------------------------------------------------------------
# Top-level pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Master configuration object.  Pass one instance to `pipeline.run()`.

    Example usage
    -------------
    >>> from resourceEstimationPipeline.config import PipelineConfig, HamlibConfig
    >>> cfg = PipelineConfig(
    ...     hamlib=HamlibConfig(hdf5_path="path/to/heis.hdf5", key_index=10),
    ... )
    """

    hamlib: HamlibConfig = field(default_factory=HamlibConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    transpile: TranspileConfig = field(default_factory=TranspileConfig)
    azure: AzureConfig = field(default_factory=AzureConfig)
    qualtran: QualtranConfig = field(default_factory=QualtranConfig)


# ---------------------------------------------------------------------------
# Convenience: a ready-to-use default config for quick experiments
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = PipelineConfig()
