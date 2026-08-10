"""
Legacy rotation gate synthesis — Solovay-Kitaev backend.

BQSKit (bqskit + bqskit-ft) is the canonical synthesis method as of the
current pipeline version.  This module is retained as a fallback for the
``synthesis_method="solovay_kitaev"`` path, which requires no extra
dependencies beyond Qiskit.

Public API
----------
synthesize_rotation(circuit, synthesis_method, epsilon) → QuantumCircuit
    Dispatcher — currently only ``"solovay_kitaev"`` is accepted here.
    For ``synthesis_method="bqskit"`` the caller should use
    ``circuit.transpile._synthesize_with_bqskit`` directly (handled
    automatically by ``transpile_to_clifford_t``).
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

log = logging.getLogger(__name__)

_ROTN_NAMES = frozenset({"rz", "rx", "ry", "r"})


def _is_rotation_gate(gate_name: str) -> bool:
    return gate_name in _ROTN_NAMES


def _angle_to_unitary_matrix(angle: float, gate_name: str) -> np.ndarray:
    if gate_name == "rz":
        return np.array(
            [[np.exp(-1j * angle / 2), 0],
             [0, np.exp(1j * angle / 2)]]
        )
    if gate_name == "rx":
        c = math.cos(angle / 2)
        s = -1j * math.sin(angle / 2)
        return np.array([[c, s], [s, c]])
    if gate_name == "ry":
        c = math.cos(angle / 2)
        s = math.sin(angle / 2)
        return np.array([[c, -s], [s, c]])
    return _angle_to_unitary_matrix(angle, "rz")


# ---------------------------------------------------------------------------
# Solovay-Kitaev backend (legacy)
# ---------------------------------------------------------------------------

def synthesize_rotation_solovay_kitaev(
    circuit: QuantumCircuit,
    epsilon: float = 1e-11,
) -> QuantumCircuit:
    """Synthesise every rotation gate into Clifford+T using Solovay-Kitaev.

    Legacy synthesis path — use ``synthesis_method="bqskit"`` for better
    T-count results via the BQSKit FT compiler.
    """
    from qiskit import QuantumCircuit as QC
    from qiskit.circuit.library import UnitaryGate
    from qiskit.transpiler import PassManager

    from ..config import PURE_CLIFFORD_T_BASIS_GATES

    out = QC(*circuit.qregs, *circuit.cregs)
    sk_pass = _make_sk_pass(epsilon)

    for instr in circuit.data:
        gate = instr.operation
        if not _is_rotation_gate(gate.name):
            out.append(gate, instr.qubits, instr.clbits)
            continue

        angle = float(gate.params[0])
        mat = _angle_to_unitary_matrix(angle, gate.name)
        sub = QC(1)
        sub.append(UnitaryGate(mat), [0])

        if sk_pass is not None:
            synth = PassManager([sk_pass]).run(sub)
            synth = _transpile_to_clifford_t(synth, list(PURE_CLIFFORD_T_BASIS_GATES))
        else:
            synth = _transpile_to_clifford_t(
                sub, list(PURE_CLIFFORD_T_BASIS_GATES), optimization_level=1
            )

        target_qubit = instr.qubits[0]
        for sub_instr in synth.data:
            out.append(sub_instr.operation, [target_qubit], [])

    return out


def _make_sk_pass(epsilon: float):
    try:
        from qiskit.transpiler.passes import SolovayKitaevDecomposition
        degree = _epsilon_to_sk_degree(epsilon)
        return SolovayKitaevDecomposition(recursion_degree=degree)
    except Exception:
        return None


def _epsilon_to_sk_degree(epsilon: float) -> int:
    if epsilon >= 1e-2:
        return 2
    if epsilon >= 1e-4:
        return 3
    if epsilon >= 1e-7:
        return 4
    return 5


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

_VALID_METHODS = frozenset({"solovay_kitaev"})


def synthesize_rotation(
    circuit: QuantumCircuit,
    synthesis_method: str = "solovay_kitaev",
    epsilon: float = 1e-11,
) -> QuantumCircuit:
    """Synthesise every rotation gate in *circuit* into Clifford+T.

    Only ``"solovay_kitaev"`` is accepted here.  For BQSKit synthesis use
    ``synthesis_method="bqskit"`` in ``TranspileConfig``; that path is handled
    directly in ``transpile_to_clifford_t``.
    """
    method_lower = synthesis_method.lower().replace("-", "_").replace(" ", "_")

    if method_lower not in _VALID_METHODS:
        if method_lower == "bqskit":
            raise ValueError(
                "synthesis_method='bqskit' must be set via TranspileConfig and is "
                "handled directly in transpile_to_clifford_t, not via this function."
            )
        if method_lower == "pygridsynth":
            raise ValueError(
                "synthesis_method='pygridsynth' has been removed. "
                "Use synthesis_method='bqskit' instead for better T-count results."
            )
        raise ValueError(
            f"Unknown synthesis method '{synthesis_method}'. "
            f"Supported: {sorted(_VALID_METHODS)}"
        )

    return synthesize_rotation_solovay_kitaev(circuit, epsilon=epsilon)


# ---------------------------------------------------------------------------
# Internal transpile helper
# ---------------------------------------------------------------------------

def _transpile_to_clifford_t(circuit, basis_gates, optimization_level=0):
    from qiskit import transpile
    return transpile(
        circuit,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
        seed_transpiler=42,
    )
