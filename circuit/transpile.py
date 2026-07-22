"""
Deterministic transpilation to a canonical Clifford+T basis.

Produces a single circuit suitable for both estimators:
  - Azure QDK reads it via OpenQASM 3 export.
  - Qualtran reads it via the CompositeBloq builder.

Transpilation pipeline (when rotation_synthesis_enabled=True)
-------------------------------------------------------------
Stage 1 — Decompose to intermediate basis
    Expand high-level gates (e.g. PauliEvolutionGate) into an intermediate
    gate set that still includes rotation gates (Rz/Rx/Ry).  No optimisation
    runs yet so rotations are preserved for combining in stage 2.

Stage 2 — Optimise with rotations in place
    Run Qiskit optimisation passes (controlled by config.optimization_level)
    while rotation gates still exist.  Consecutive rotations on the same
    qubit can be merged/cancelled here before the expensive synthesis step.

Stage 3 — Synthesise arbitrary rotations into Clifford+T
    Replace every Rz/Rx/Ry gate with a Clifford+T approximation using
    Qiskit's built-in synthesis.  Precision is controlled by
    config.rotation_synthesis_epsilon.  After this stage no rotation gates
    remain.

Stage 4 — Final cleanup to pure Clifford+T
    Run a lightweight transpile pass to normalise any residual non-Clifford+T
    gates that Qiskit may have introduced during synthesis.  Validate that no
    rotation gates remain.

Public API
----------
transpile_to_clifford_t(circuit, config) -> QuantumCircuit
pre_synthesize_rz(circuit, epsilon) -> QuantumCircuit
circuit_to_qasm(circuit) -> str
circuit_stats(circuit) -> dict
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.qasm3 import Exporter

from ..config import (
    INTERMEDIATE_BASIS_GATES,
    PURE_CLIFFORD_T_BASIS_GATES,
    TranspileConfig,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level basis constants
# ---------------------------------------------------------------------------

_INTERMEDIATE_BASIS: list = list(INTERMEDIATE_BASIS_GATES)
"""Stage 1–2 basis: keeps rotation gates for optimisation."""

_PURE_CLIFFORD_T_BASIS: list = list(PURE_CLIFFORD_T_BASIS_GATES)
"""Stage 3–4 basis: pure Clifford+T, no arbitrary rotations."""

_ROTN_NAMES: frozenset = frozenset({"rz", "rx", "ry", "r"})
"""Gate names that carry an arbitrary rotation angle."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _angle_to_unitary_matrix(angle: float, gate_name: str) -> np.ndarray:
    """Convert a single-qubit rotation angle to its 2×2 unitary matrix."""
    if gate_name == "rz":
        return np.array([[np.exp(-1j * angle / 2), 0],
                         [0,  np.exp( 1j * angle / 2)]])
    if gate_name == "rx":
        c = math.cos(angle / 2)
        s = -1j * math.sin(angle / 2)
        return np.array([[c, s], [s, c]])
    if gate_name == "ry":
        c =  math.cos(angle / 2)
        s =  math.sin(angle / 2)
        return np.array([[c, -s], [s, c]])
    # generic "r" — treat as Rz
    return _angle_to_unitary_matrix(angle, "rz")


def _count_rotations(circuit: QuantumCircuit) -> Dict[str, int]:
    """Return per-name counts for all rotation gates in the circuit."""
    ops = dict(circuit.count_ops())
    return {name: ops.get(name, 0) for name in _ROTN_NAMES}


def _log_rotation_counts(label: str, counts: Dict[str, int]) -> None:
    total = sum(counts.values())
    detail = ", ".join(f"{n}={v}" for n, v in counts.items() if v)
    log.info("[transpile] %s: %d rotation gate(s) (%s)", label, total, detail or "none")
    print(f"[transpile] {label}: {total} rotation gate(s) ({detail or 'none'})")


def _validate_no_rotations(circuit: QuantumCircuit) -> None:
    """Raise RuntimeError if any rotation gate survives into the final circuit."""
    counts = _count_rotations(circuit)
    remaining = sum(counts.values())
    if remaining > 0:
        detail = ", ".join(f"{n}={v}" for n, v in counts.items() if v)
        raise RuntimeError(
            f"[transpile] Stage 4 validation failed: {remaining} rotation gate(s) "
            f"remain after synthesis ({detail}). "
            "Check that pre_synthesize_rz() handled all rotation gate types."
        )
    log.info("[transpile] Validation passed: no rotation gates in final circuit.")


# ---------------------------------------------------------------------------
# Stage 3: rotation pre-synthesis
# ---------------------------------------------------------------------------

def pre_synthesize_rz(
    circuit: QuantumCircuit,
    epsilon: float = 1e-11,
    synthesis_method: str = "solovay_kitaev",
) -> QuantumCircuit:
    """
    Decompose every arbitrary rotation gate into Clifford+T.

    Iterates the circuit instruction-by-instruction and rebuilds a new
    circuit, replacing each rotation gate with its synthesised Clifford+T
    expansion in place.  Non-rotation gates are copied verbatim.

    This function is now a thin dispatcher around :func:`rotation_synthesis.synthesize_rotation`.
    The ``synthesis_method`` parameter selects which backend to use:

    - ``"solovay_kitaev"`` (default) — uses Qiskit's Solovay-Kitaev pass.
      **This is the existing path and its behaviour is identical to prior versions.**
    - ``"pygridsynth"`` — uses pygridsynth for optimal or near-optimal exact
      Clifford+T synthesis.  Requires ``pip install pygridsynth``.

    Parameters
    ----------
    circuit : QuantumCircuit
        May contain any mixture of Clifford+T and rotation gates.
    epsilon : float
        Approximation precision hint.  Interpretation depends on backend:
        Solovay-Kitaev → upper bound on diamond-norm error per gate.
        pygridsynth     → maximum distance from nearest Clifford+T grid point.
    synthesis_method : str
        Which synthesis algorithm to use (see above).

    Returns
    -------
    QuantumCircuit
        Equivalent circuit with all rotation gates replaced by Clifford+T
        sequences.  Gate ordering is preserved.
    """
    # Import the unified dispatcher — lazy to avoid hard dependency.
    from .rotation_synthesis import synthesize_rotation as _synth

    return _synth(circuit, synthesis_method=synthesis_method, epsilon=epsilon)


def _make_sk_pass(epsilon: float):
    """Return a SolovayKitaevDecomposition pass instance, or None if unavailable."""
    try:
        from qiskit.transpiler.passes import SolovayKitaevDecomposition
        degree = _epsilon_to_sk_degree(epsilon)
        return SolovayKitaevDecomposition(recursion_degree=degree)
    except (ImportError, Exception):
        return None


def _epsilon_to_sk_degree(epsilon: float) -> int:
    """Map approximation epsilon to a Solovay-Kitaev recursion degree."""
    if epsilon >= 1e-2:
        return 2
    if epsilon >= 1e-4:
        return 3
    if epsilon >= 1e-7:
        return 4
    return 5


# ---------------------------------------------------------------------------
# Main transpilation entry point
# ---------------------------------------------------------------------------

def transpile_to_clifford_t(
    circuit: QuantumCircuit,
    config: TranspileConfig,
) -> QuantumCircuit:
    """
    Transpile a Qiskit circuit to a pure Clifford+T basis via a 4-stage pipeline.

    Stage 1 — Decompose to intermediate basis (rotations kept)
    Stage 2 — Optimise while rotations exist (combine/cancel Rz chains)
    Stage 3 — Synthesise every Rz/Rx/Ry into Clifford+T
    Stage 4 — Final cleanup + validation (pure Clifford+T output)

    When ``rotation_synthesis_enabled`` is False (or legacy
    ``synthesis_strategy == "passthrough"``), the pipeline collapses to a
    single ``transpile()`` call using ``config.basis_gates``.

    Parameters
    ----------
    circuit : QuantumCircuit
        Input circuit (may contain high-level gates such as PauliEvolutionGate).
    config  : TranspileConfig

    Returns
    -------
    QuantumCircuit in a pure Clifford+T basis with no arbitrary rotations.
    """
    # Resolve legacy synthesis_strategy field alongside the new boolean flag.
    synthesis_enabled = config.rotation_synthesis_enabled
    if config.synthesis_strategy == "passthrough":
        synthesis_enabled = False

    if not synthesis_enabled:
        # Passthrough: single-stage transpile, rotations pass through verbatim.
        log.info("[transpile] Passthrough mode: single-stage transpile.")
        return transpile(
            circuit,
            basis_gates=config.basis_gates,
            optimization_level=config.optimization_level,
            seed_transpiler=config.seed_transpiler,
        )

    # ── Stage 1: Decompose to intermediate basis (keeps rotation gates) ───────
    log.info("[transpile] Stage 1: decompose to intermediate basis (rotations kept).")
    print("[transpile] Stage 1: decomposing to intermediate basis...")
    s1 = transpile(
        circuit,
        basis_gates=_INTERMEDIATE_BASIS,
        optimization_level=0,       # no optimisation yet — just decompose
        seed_transpiler=config.seed_transpiler,
    )

    # ── Stage 2: Optimise while rotations still exist ────────────────────────
    log.info("[transpile] Stage 2: optimising with rotations in place (level=%d).",
             config.optimization_level)
    print(f"[transpile] Stage 2: optimising (level={config.optimization_level}) "
          "with rotations in place...")
    s2 = transpile(
        s1,
        basis_gates=_INTERMEDIATE_BASIS,
        optimization_level=config.optimization_level,
        seed_transpiler=config.seed_transpiler,
    )

    # ── Stage 3: Synthesise arbitrary rotations into Clifford+T ──────────────
    pre_counts = _count_rotations(s2)
    _log_rotation_counts("Before synthesis", pre_counts)

    log.info("[transpile] Stage 3: synthesising rotations into Clifford+T "
             "(epsilon=%.2e, method=%s).", config.rotation_synthesis_epsilon,
             config.synthesis_method)
    print(f"[transpile] Stage 3: synthesising rotations "
          f"(epsilon={config.rotation_synthesis_epsilon:.2e}, "
          f"method={config.synthesis_method})...")

    # Resolve pygridsynth_precision → fallback to rotation_synthesis_epsilon.
    syn_epsilon = (
        config.pygridsynth_precision
        if config.pygridsynth_precision is not None
        else config.rotation_synthesis_epsilon
    )
    s3 = pre_synthesize_rz(
        s2,
        epsilon=syn_epsilon,
        synthesis_method=config.synthesis_method,
    )

    post_counts = _count_rotations(s3)
    _log_rotation_counts("After synthesis", post_counts)
    _print_gate_summary(s3)

    # ── Stage 4: Final cleanup to pure Clifford+T + validation ───────────────
    log.info("[transpile] Stage 4: final cleanup to pure Clifford+T.")
    print("[transpile] Stage 4: final cleanup to pure Clifford+T...")
    s4 = transpile(
        s3,
        basis_gates=_PURE_CLIFFORD_T_BASIS,
        optimization_level=0,       # no further optimisation — just normalise
        seed_transpiler=config.seed_transpiler,
    )
    _validate_no_rotations(s4)

    ops = dict(s4.count_ops())
    print(
        f"[transpile] Done. T={ops.get('t', 0)+ops.get('tdg', 0)}  "
        f"CX={ops.get('cx', 0)}  "
        f"Clifford(H/S)={ops.get('h', 0)+ops.get('s', 0)+ops.get('sdg', 0)}  "
        f"depth={s4.depth()}"
    )
    return s4


def _print_gate_summary(circuit: QuantumCircuit) -> None:
    """Print post-synthesis gate counts for diagnostics."""
    ops = dict(circuit.count_ops())
    t_count = ops.get("t", 0) + ops.get("tdg", 0)
    clifford_count = sum(ops.get(g, 0) for g in ("h", "s", "sdg", "x", "y", "z"))
    cx_count = ops.get("cx", 0)
    rot_remaining = sum(ops.get(g, 0) for g in _ROTN_NAMES)
    unsupported = {k: v for k, v in ops.items()
                   if k not in {*_PURE_CLIFFORD_T_BASIS, "measure", "barrier", "reset"}}
    print(
        f"[transpile]   Post-synthesis: T={t_count}  Clifford={clifford_count}  "
        f"CX={cx_count}  rotations_remaining={rot_remaining}  "
        f"unsupported={unsupported or 'none'}"
    )


# ---------------------------------------------------------------------------
# OpenQASM 3 export  (Azure QDK front-end)
# ---------------------------------------------------------------------------

def circuit_to_qasm(circuit: QuantumCircuit) -> str:
    """
    Export a transpiled QuantumCircuit to an OpenQASM 3 string.

    Source: estimator/circuitBuilderGeneralized.ipynb — circuit_to_qasm helper.

    Parameters
    ----------
    circuit : QuantumCircuit  (should already be transpiled to the canonical basis)

    Returns
    -------
    str  — OpenQASM 3 source text, ready for ``OpenQASMApplication``.
    """
    return Exporter().dumps(circuit)


# ---------------------------------------------------------------------------
# Utility: gate-count summary
# ---------------------------------------------------------------------------

def compute_t_depth(circuit: QuantumCircuit) -> int:
    """
    Compute T-gate depth using Qiskit DAG layer analysis.

    Returns the number of circuit layers that contain at least one T or
    T-dagger gate (i.e., the T-gate critical-path depth).
    """
    from qiskit.converters import circuit_to_dag

    dag = circuit_to_dag(circuit)
    t_gate_names = {"t", "tdg"}
    return sum(
        1
        for layer in dag.layers()
        if any(node.op.name in t_gate_names for node in layer["graph"].op_nodes())
    )


def circuit_stats(circuit: QuantumCircuit) -> Dict[str, Any]:
    """
    Return a summary dictionary of circuit statistics.

    Parameters
    ----------
    circuit : QuantumCircuit

    Returns
    -------
    dict with keys:
      num_qubits, depth, t_depth, total_gates, gate_counts (dict),
      t_count, clifford_count, cx_count, rotation_count, rz_count
    """
    ops = dict(circuit.count_ops())
    t_gates = {"t", "tdg"}
    clifford_gates = {"h", "s", "sdg", "x", "y", "z", "cx", "cz", "swap", "ccx"}
    t_count = sum(ops.get(g, 0) for g in t_gates)
    clifford_count = sum(ops.get(g, 0) for g in clifford_gates)
    rotation_count = sum(ops.get(g, 0) for g in _ROTN_NAMES)

    return {
        "num_qubits":     circuit.num_qubits,
        "depth":          circuit.depth(),
        "t_depth":        compute_t_depth(circuit),
        "total_gates":    sum(ops.values()),
        "gate_counts":    ops,
        "t_count":        t_count,
        "clifford_count": clifford_count,
        "cx_count":       ops.get("cx", 0),
        "rotation_count": rotation_count,
        "rz_count":       ops.get("rz", 0),
    }
