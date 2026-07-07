"""
Common estimator interface and result dataclass.

Both `azure.py` and `qualtran.py` return `EstimationResult` and implement
the `Estimator` protocol so comparison code never depends on estimator internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, runtime_checkable, Protocol

from qiskit import QuantumCircuit


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

@dataclass
class EstimationResult:
    """
    Estimator-independent container for resource estimates.

    Fields that a given estimator cannot compute are left as `None`;
    the comparison layer renders them as 'N/A' in tables.
    """

    estimator_name: str
    """Human-readable name, e.g. 'Azure QDK' or 'Qualtran (d=17)'."""

    # ── Logical / algorithmic counts ─────────────────────────────────────────
    logical_qubits: Optional[int] = None
    """Number of logical (algorithm) qubits."""

    t_count: Optional[int] = None
    """Total T-gate count (including those synthesised from arbitrary Rz)."""

    t_depth: Optional[int] = None
    """T-gate depth (T gates on the critical path)."""

    clifford_count: Optional[int] = None
    """Total Clifford gate count (H, S, CX, …)."""

    rotation_count: Optional[int] = None
    """Number of arbitrary Rz rotations (before T-gate synthesis)."""

    toffoli_count: Optional[int] = None
    """Toffoli / CCZ gate count."""

    measurement_count: Optional[int] = None
    """Number of mid-circuit or final measurements."""

    # ── Physical resource counts ──────────────────────────────────────────────
    physical_qubits: Optional[int] = None
    """Total physical qubit count (compute + factory + memory)."""

    physical_compute_qubits: Optional[int] = None
    """Physical qubits used for the logical compute register."""

    physical_factory_qubits: Optional[int] = None
    """Physical qubits used by the magic-state factory."""

    physical_memory_qubits: Optional[int] = None
    """Physical qubits used for logical memory (if reported)."""

    # ── Timing ───────────────────────────────────────────────────────────────
    runtime_seconds: Optional[float] = None
    """Estimated wall-clock runtime in seconds."""

    # ── Error budget ─────────────────────────────────────────────────────────
    error_budget: Optional[float] = None
    """Total error budget supplied to the estimator."""

    logical_error_rate: Optional[float] = None
    """Estimated total logical failure probability."""

    # ── QEC / architecture ───────────────────────────────────────────────────
    code_distance: Optional[int] = None
    """Surface-code data-block distance."""

    factory_count: Optional[str] = None
    """Description of the magic-state factory configuration, e.g. '4×T'."""

    t_per_rotation: Optional[int] = None
    """Number of T gates used to synthesise each arbitrary Rz rotation."""

    # ── Assumptions ──────────────────────────────────────────────────────────
    algorithm_assumptions: Optional[str] = None
    """Free-text description of algorithmic assumptions."""

    architecture_assumptions: Optional[str] = None
    """Free-text description of hardware / architecture assumptions."""

    # ── Raw estimator output (opaque) ────────────────────────────────────────
    raw: Optional[Any] = field(default=None, repr=False)
    """The unmodified object returned by the underlying estimator library."""

    extra: Dict[str, Any] = field(default_factory=dict, repr=False)
    """Catch-all for estimator-specific fields not covered above."""

    # ──────────────────────────────────────────────────────────────────────────
    def as_dict(self) -> Dict[str, Any]:
        """Return a flat {metric: value} dictionary for table rendering."""
        return {
            "Estimator": self.estimator_name,
            "Logical qubits": self.logical_qubits,
            "Physical qubits (total)": self.physical_qubits,
            "Physical compute qubits": self.physical_compute_qubits,
            "Physical factory qubits": self.physical_factory_qubits,
            "Physical memory qubits": self.physical_memory_qubits,
            "T count": self.t_count,
            "T depth": self.t_depth,
            "Clifford count": self.clifford_count,
            "Rotation count (arb. Rz)": self.rotation_count,
            "Toffoli count": self.toffoli_count,
            "Measurement count": self.measurement_count,
            "Runtime (s)": self.runtime_seconds,
            "Error budget": self.error_budget,
            "Logical error rate": self.logical_error_rate,
            "Code distance": self.code_distance,
            "Factory config": self.factory_count,
            "T gates per rotation": self.t_per_rotation,
            "Algorithm assumptions": self.algorithm_assumptions,
            "Architecture assumptions": self.architecture_assumptions,
        }


# ---------------------------------------------------------------------------
# Estimator interface
# ---------------------------------------------------------------------------

@runtime_checkable
class Estimator(Protocol):
    """
    Protocol (structural interface) for resource estimators.

    To add a new estimator, create a module in `estimators/` that defines a
    class or callable satisfying this protocol — no inheritance required.
    """

    def estimate(
        self,
        circuit: QuantumCircuit,
        config: Any,
    ) -> EstimationResult:
        """
        Run resource estimation on a transpiled Clifford+T circuit.

        Parameters
        ----------
        circuit : QuantumCircuit
            The canonical Clifford+T circuit produced by `circuit.transpile`.
        config  : PipelineConfig or sub-config
            Full pipeline configuration (estimator reads its own sub-config).

        Returns
        -------
        EstimationResult
            Estimator-independent result object.  Fields the estimator cannot
            compute must be left as `None`.
        """
        ...
