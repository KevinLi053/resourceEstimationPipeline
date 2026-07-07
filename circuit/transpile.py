"""
Deterministic transpilation to a canonical Clifford+T basis.

Produces a single circuit suitable for both estimators:
  - Azure QDK reads it via OpenQASM 3 export.
  - Qualtran reads it via the CompositeBloq builder.

Reuses transpilation logic from:
  - estimator/analysis/hamlib.ipynb
  - estimator/circuitBuilderGeneralized.ipynb  (circuit_to_qasm helper)

Public API
----------
transpile_to_clifford_t(circuit, config) -> QuantumCircuit
    Transpile to the canonical basis defined in TranspileConfig.

circuit_to_qasm(circuit) -> str
    Export a transpiled circuit to an OpenQASM 3 string (for Azure).

circuit_stats(circuit) -> dict
    Return a summary dict of gate counts and depth.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from qiskit import QuantumCircuit, transpile
from qiskit.qasm3 import Exporter

from resourceEstimationPipeline.config import TranspileConfig


# ---------------------------------------------------------------------------
# Main transpilation step
# ---------------------------------------------------------------------------

def transpile_to_clifford_t(
    circuit: QuantumCircuit,
    config: TranspileConfig,
) -> QuantumCircuit:
    """
    Transpile a Qiskit circuit to the canonical Clifford+T basis.

    Source: estimator/analysis/hamlib.ipynb and circuitBuilderGeneralized.ipynb.

    Uses a deterministic Qiskit transpiler pass with a fixed random seed so
    repeated runs always produce the same gate sequence.

    Parameters
    ----------
    circuit : QuantumCircuit
        Input circuit (may contain high-level gates such as PauliEvolutionGate).
    config  : TranspileConfig
        Controls basis gates, optimization level, and random seed.

    Returns
    -------
    QuantumCircuit in the requested Clifford+T basis.
    """
    # For PauliEvolutionGate-based circuits we must decompose before transpiling
    # (transpile will handle this automatically, but an explicit decompose call
    # makes the gate count visible for debugging).
    return transpile(
        circuit,
        basis_gates=config.basis_gates,
        optimization_level=config.optimization_level,
        seed_transpiler=config.seed_transpiler,
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

def circuit_stats(circuit: QuantumCircuit) -> Dict[str, Any]:
    """
    Return a summary dictionary of circuit statistics.

    Parameters
    ----------
    circuit : QuantumCircuit

    Returns
    -------
    dict with keys:
      num_qubits, depth, total_gates, gate_counts (dict), t_count, clifford_count
    """
    ops = dict(circuit.count_ops())
    t_gates = {"t", "tdg"}
    clifford_gates = {"h", "s", "sdg", "x", "y", "z", "cx", "cz", "swap", "ccx"}
    t_count = sum(ops.get(g, 0) for g in t_gates)
    clifford_count = sum(ops.get(g, 0) for g in clifford_gates)

    return {
        "num_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "total_gates": sum(ops.values()),
        "gate_counts": ops,
        "t_count": t_count,
        "clifford_count": clifford_count,
        "rz_count": ops.get("rz", 0),
    }
