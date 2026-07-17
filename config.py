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
    Currently supported: 'CCZ2T' (Gidney-Fowler CCZ-to-T factory) and 'FifteenToOne.
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
