"""
Common estimator interface and result dataclass.

Both `azure.py` and `qualtran.py` return `EstimationResult` and implement
the `Estimator` protocol so comparison code never depends on estimator internals.
"""
from __future__ import annotations

import math
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

    Sections
    --------
    Logical / algorithmic counts  – circuit-level gate and qubit counts
    Physical resource counts      – hardware qubit breakdown
    Timing                        – wall-clock runtime
    Error budget & QEC            – error tolerance and code parameters
    Factory                       – magic-state factory model and config
    Rotation synthesis            – Rz→T synthesis parameters
    Physical parameters           – estimator hardware assumptions
    Assumptions (text)            – free-text notes on model choices
    Raw / extra                   – opaque estimator output for debugging
    """

    estimator_name: str
    """Human-readable name, e.g. 'Azure QDK' or 'Qualtran (d=17)'."""

    # ── Logical / algorithmic counts ─────────────────────────────────────────
    logical_qubits: Optional[int] = None
    """Number of logical (algorithm) qubits."""

    logical_depth: Optional[int] = None
    """Logical circuit depth (gate layers, not QEC rounds)."""

    logical_cycles: Optional[int] = None
    """Total number of QEC syndrome-measurement rounds for the algorithm."""

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

    # ── Error budget & QEC ───────────────────────────────────────────────────
    error_budget: Optional[float] = None
    """Total error budget supplied to the estimator."""

    logical_error_rate: Optional[float] = None
    """Estimated total logical failure probability."""

    code_distance: Optional[int] = None
    """Surface-code data-block distance."""

    # ── Factory ──────────────────────────────────────────────────────────────
    factory_type: Optional[str] = None
    """Magic-state factory model name, e.g. 'Litinski19', 'CCZ2T'."""

    factory_count: Optional[str] = None
    """Description of the magic-state factory configuration, e.g. '4×T'."""

    # ── Rotation synthesis ───────────────────────────────────────────────────
    t_per_rotation: Optional[int] = None
    """Number of T gates used to synthesise each arbitrary Rz rotation."""

    rotation_synthesis_precision: Optional[float] = None
    """Target synthesis error per Rz rotation (rz_eps / ε_synth)."""

    # ── Physical parameters (estimator hardware assumptions) ─────────────────
    physical_error_rate: Optional[float] = None
    """Physical gate error rate assumed by the estimator."""

    cycle_time_us: Optional[float] = None
    """Surface-code cycle time in microseconds."""

    gate_time_ns: Optional[float] = None
    """Single/two-qubit gate time in nanoseconds (GateBased model)."""

    measurement_time_ns: Optional[float] = None
    """Measurement time in nanoseconds (GateBased model)."""

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
    def derived_metrics(self) -> Dict[str, Optional[float]]:
        """
        Compute secondary metrics from primary fields.

        All results are None when the required inputs are missing.
        Approximations are noted inline.
        """
        def _safe_div(num, den):
            if num is None or den is None or den == 0:
                return None
            return num / den

        # physical_qubits_per_logical_qubit: total overhead per algorithmic qubit
        phys_per_log = _safe_div(self.physical_qubits, self.logical_qubits)

        # factory_qubit_fraction: share of physical qubits in the factory
        factory_frac = _safe_div(self.physical_factory_qubits, self.physical_qubits)

        # runtime_per_T_gate: amortised wall-clock cost per T state
        runtime_per_t = _safe_div(self.runtime_seconds, self.t_count)

        # T_per_logical_qubit: T gates per logical qubit (algorithm complexity proxy)
        t_per_lq = _safe_div(self.t_count, self.logical_qubits)

        # logical_cycles_per_T_gate: QEC rounds per T gate (execution density)
        # Approximate: valid when T gates dominate the schedule.
        cycles_per_t = _safe_div(self.logical_cycles, self.t_count)

        # physical_qubits_per_T_state: amortised factory qubits per T state consumed.
        # Approximate: treats factory qubits as statically allocated for the full run.
        phys_per_t = _safe_div(self.physical_factory_qubits, self.t_count)

        return {
            "physical_qubits_per_logical_qubit": phys_per_log,
            "factory_qubit_fraction": factory_frac,
            "runtime_per_T_gate": runtime_per_t,
            "T_per_logical_qubit": t_per_lq,
            "logical_cycles_per_T_gate": cycles_per_t,
            "physical_qubits_per_T_state": phys_per_t,
        }

    def as_dict(self) -> Dict[str, Any]:
        """Return a flat {metric: value} dictionary for table rendering."""
        dm = self.derived_metrics()
        return {
            # ── Identity ──────────────────────────────────────────────────────
            "Estimator": self.estimator_name,

            # ── Logical / algorithmic ─────────────────────────────────────────
            "Logical qubits": self.logical_qubits,
            "Logical depth": self.logical_depth,
            "Logical cycles": self.logical_cycles,

            # ── Gate counts ───────────────────────────────────────────────────
            "T count": self.t_count,
            "T depth": self.t_depth,
            "Clifford count": self.clifford_count,
            "Rotation count (arb. Rz)": self.rotation_count,
            "Toffoli count": self.toffoli_count,
            "Measurement count": self.measurement_count,

            # ── Physical resources ────────────────────────────────────────────
            "Physical qubits (total)": self.physical_qubits,
            "Physical compute qubits": self.physical_compute_qubits,
            "Physical factory qubits": self.physical_factory_qubits,
            "Physical memory qubits": self.physical_memory_qubits,

            # ── Timing ────────────────────────────────────────────────────────
            "Runtime (s)": self.runtime_seconds,

            # ── Error & QEC ───────────────────────────────────────────────────
            "Error budget": self.error_budget,
            "Logical error rate": self.logical_error_rate,
            "Code distance": self.code_distance,

            # ── Factory ───────────────────────────────────────────────────────
            "Factory type": self.factory_type,
            "Factory config": self.factory_count,

            # ── Rotation synthesis ────────────────────────────────────────────
            "T gates per rotation": self.t_per_rotation,
            "Rotation synthesis precision (ε)": self.rotation_synthesis_precision,

            # ── Physical parameters (assumptions) ─────────────────────────────
            "Physical error rate": self.physical_error_rate,
            "Cycle time (µs)": self.cycle_time_us,
            "Gate time (ns)": self.gate_time_ns,
            "Measurement time (ns)": self.measurement_time_ns,

            # ── Derived metrics ───────────────────────────────────────────────
            "Physical qubits per logical qubit": dm["physical_qubits_per_logical_qubit"],
            "Factory qubit fraction": dm["factory_qubit_fraction"],
            "Runtime per T gate (s)": dm["runtime_per_T_gate"],
            "T gates per logical qubit": dm["T_per_logical_qubit"],
            "Logical cycles per T gate": dm["logical_cycles_per_T_gate"],
            "Physical qubits per T state": dm["physical_qubits_per_T_state"],

            # ── Assumptions ───────────────────────────────────────────────────
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
