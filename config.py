"""
Central configuration for the resource-estimation pipeline.

Changing parameters for either estimator requires editing only this file.
All tunable knobs are grouped by concern: Hamiltonian loading, circuit
construction, transpilation, Azure QDK, and Qualtran.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


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

# Canonical Clifford+T basis accepted by both Azure QDK and Qualtran.
# This is the intersection of both estimators' gate sets and is the
# standard fault-tolerant gate set.
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
    Gate set to transpile into. Both estimators share this canonical basis
    so they operate on the same circuit. Override with QUALTRAN_EXTENDED_BASIS_GATES
    to let Qualtran use its native cz/ccx/swap bloqs instead.
    """

    optimization_level: int = 1
    """
    Qiskit transpiler optimization level.
    0 = decompose only (preserves gate counts exactly).
    1 = light optimisation (default, balances depth vs. compilation time).
    2–3 = heavier optimisation (slower build, smaller circuit).
    """

    seed_transpiler: Optional[int] = 42
    """Random seed for the transpiler (determinism)."""


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
    """Single/two-qubit gate time in nanoseconds."""

    measurement_time_ns: float = 100.0
    """Measurement time in nanoseconds."""

    # QEC: surface-code ISA
    surface_code_distances: Optional[List[int]] = None
    """
    Explicit code distances to sweep. None = let qdk.qre sweep automatically
    via SurfaceCode.q().
    """

    # Magic-state factory
    factory_type: str = "Litinski19"
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

    Reference implementations in: qualtranEstimator/qualtranCircuitBuilder.ipynb
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

    cycle_time_us: float = 1.0
    """Surface-code cycle time in microseconds."""

    # Rotation synthesis
    rz_eps: float = 1e-11
    """
    Synthesis precision for arbitrary Rz gates when converting them to
    Clifford+T sequences inside the CompositeBloq builder.
    """

    # Magic-state factory
    factory_type: str = "CCZ2T"
    """
    Qualtran magic-state factory model used in the custom cost-model path.
    Currently supported: 'CCZ2T' (Gidney-Fowler CCZ-to-T factory).
    Only takes effect when use_gidney_fowler=False and use_beverland=False.
    """

    # Decomposition / estimation precision
    use_gidney_fowler: bool = False
    """
    If True, use the Gidney-Fowler CCZ2T factory model.
    If False, construct a custom PhysicalCostModel from `phys_err` and `cycle_time_us`.
    """

    use_beverland: bool = True

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
