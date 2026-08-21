"""
Deterministic transpilation to a canonical Clifford+T basis.

Produces a single circuit suitable for both estimators:
  - Azure QDK reads it via OpenQASM 3 export.
  - Qualtran reads it via the CompositeBloq builder.

Transpilation pipeline (when rotation_synthesis_enabled=True)
-------------------------------------------------------------
BQSKit path (default, synthesis_method="bqskit")
    Compile the circuit holistically to Clifford+T using BQSKit's
    CliffordTModel (from bqskit.ft).  This replaces the old staged Qiskit
    pipeline (stages 1-3) with a single BQSKit compile call, followed by a
    final Qiskit cleanup pass (stage 4) to normalise any residual gates.

Legacy Qiskit path (synthesis_method="solovay_kitaev")
    Stage 1 — Decompose to intermediate basis (rotations kept).
    Stage 2 — Optimise with rotations in place.
    Stage 3 — Synthesise Rz/Rx/Ry into Clifford+T via Solovay-Kitaev.
    Stage 4 — Final cleanup to pure Clifford+T + validation.

Public API
----------
transpile_to_clifford_t(circuit, config) -> QuantumCircuit
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
# BQSKit synthesis (primary path)
# ---------------------------------------------------------------------------

def _synthesize_with_bqskit(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Compile a Qiskit circuit to Clifford+T using BQSKit FT.

    Uses bqskit.ft.CliffordTModel with T and Tdg as the only non-Clifford
    gates.  This replaces the old three-stage Qiskit rotation-synthesis
    pipeline with a single holistic BQSKit compile call.

    The output may contain SX (√X) gates; the Stage 4 Qiskit transpile pass
    decomposes those into the pure Clifford+T basis.
    """
    try:
        from bqskit import compile as bqskit_compile
        from bqskit.ft import CliffordTModel
        from bqskit.ft.cliffordt.cliffordtgates import t_gates
        from bqskit.ext import qiskit_to_bqskit, bqskit_to_qiskit
        from bqskit.ir.gates.constant.t import TGate
        from bqskit.ir.gates.constant.tdg import TdgGate
    except ImportError as exc:
        raise ImportError(
            "synthesis_method='bqskit' requires the 'bqskit' and 'bqskit-ft' packages. "
            "Install with: pip install bqskit bqskit-ft"
        ) from exc

    log.info("[transpile/bqskit] Converting Qiskit circuit to BQSKit format.")
    print("[transpile/bqskit] Converting to BQSKit format...")
    bq_circuit = qiskit_to_bqskit(circuit)

    log.info("[transpile/bqskit] Pre-compile gate counts: %s", bq_circuit.gate_counts)
    print(f"[transpile/bqskit] Pre-compile gate counts: {bq_circuit.gate_counts}")

    model = CliffordTModel(
        num_qudits=bq_circuit.num_qudits,
        non_clifford_gates=t_gates,
    )

    log.info("[transpile/bqskit] Compiling with CliffordTModel (BQSKit FT).")
    print("[transpile/bqskit] Compiling with BQSKit CliffordTModel...")
    ft_circuit = bqskit_compile(bq_circuit, model)

    log.info("[transpile/bqskit] Post-compile gate set: %s", ft_circuit.gate_set)
    print(f"[transpile/bqskit] Post-compile gate set: {ft_circuit.gate_set}")

    return bqskit_to_qiskit(ft_circuit)


# ---------------------------------------------------------------------------
# Legacy Solovay-Kitaev rotation pre-synthesis
# ---------------------------------------------------------------------------

def _pre_synthesize_rz_solovay_kitaev(
    circuit: QuantumCircuit,
    epsilon: float = 1e-11,
) -> QuantumCircuit:
    """Synthesise rotation gates into Clifford+T using Solovay-Kitaev (legacy path)."""
    from .rotation_synthesis import synthesize_rotation as _synth
    return _synth(circuit, synthesis_method="solovay_kitaev", epsilon=epsilon)


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
        print("[transpile] Passthrough mode: preserving rotation gates (synthesis disabled).")
        result = transpile(
            circuit,
            basis_gates=config.basis_gates,
            optimization_level=config.optimization_level,
            seed_transpiler=config.seed_transpiler,
        )
        counts = _count_rotations(result)
        _log_rotation_counts("Passthrough result (rotations preserved)", counts)
        n_rots = sum(counts.values())
        n_t = dict(result.count_ops()).get("t", 0) + dict(result.count_ops()).get("tdg", 0)
        print(
            f"[transpile] Passthrough done: "
            f"rotation_gates={n_rots}  T_gates={n_t}  depth={result.depth()}"
        )
        if n_rots == 0:
            log.warning(
                "[transpile] Passthrough mode produced 0 rotation gates. "
                "The input circuit may already be in a Clifford+T basis, or all "
                "rotations were cancelled during optimization. This is expected only "
                "if the Hamiltonian has no non-Clifford Trotter terms."
            )
            print(
                "[transpile] WARNING: 0 rotation gates in passthrough output. "
                "If you expected Rz gates, check basis_gates and optimization_level."
            )
        return result

    method = config.synthesis_method.lower().strip()

    # ── BQSKit path (default) ─────────────────────────────────────────────────
    if method == "bqskit":
        log.info("[transpile] BQSKit path: compiling directly to Clifford+T.")
        print("[transpile] BQSKit: compiling circuit to Clifford+T...")
        s3 = _synthesize_with_bqskit(circuit)

        post_counts = _count_rotations(s3)
        _log_rotation_counts("After BQSKit synthesis", post_counts)
        _print_gate_summary(s3)

    # ── Legacy Solovay-Kitaev path ────────────────────────────────────────────
    elif method == "solovay_kitaev":
        # Stage 1: Decompose to intermediate basis (keeps rotation gates).
        log.info("[transpile] Stage 1: decompose to intermediate basis (rotations kept).")
        print("[transpile] Stage 1: decomposing to intermediate basis...")
        s1 = transpile(
            circuit,
            basis_gates=_INTERMEDIATE_BASIS,
            optimization_level=0,
            seed_transpiler=config.seed_transpiler,
        )

        # Stage 2: Optimise while rotations still exist.
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

        # Stage 3: Synthesise arbitrary rotations into Clifford+T.
        pre_counts = _count_rotations(s2)
        _log_rotation_counts("Before synthesis", pre_counts)

        log.info("[transpile] Stage 3: synthesising rotations (epsilon=%.2e, method=solovay_kitaev).",
                 config.rotation_synthesis_epsilon)
        print(f"[transpile] Stage 3: synthesising rotations "
              f"(epsilon={config.rotation_synthesis_epsilon:.2e}, method=solovay_kitaev)...")

        s3 = _pre_synthesize_rz_solovay_kitaev(s2, epsilon=config.rotation_synthesis_epsilon)

        post_counts = _count_rotations(s3)
        _log_rotation_counts("After synthesis", post_counts)
        _print_gate_summary(s3)

    else:
        raise ValueError(
            f"Unknown synthesis_method '{config.synthesis_method}'. "
            "Supported values: 'bqskit' (default), 'solovay_kitaev' (legacy)."
        )

    # ── Stage 4: Final cleanup to pure Clifford+T + validation ───────────────
    log.info("[transpile] Stage 4: final cleanup to pure Clifford+T.")
    print("[transpile] Stage 4: final cleanup to pure Clifford+T...")
    s4 = transpile(
        s3,
        basis_gates=_PURE_CLIFFORD_T_BASIS,
        optimization_level=0,
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
    clifford_gates = {"cx", "cz", "h", "s", "sdg", "sx", "swap", "x", "y", "z"}
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
