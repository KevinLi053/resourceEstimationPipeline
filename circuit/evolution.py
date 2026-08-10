"""
Hamiltonian time evolution circuit via Qiskit's PauliEvolutionGate.

Public API
----------
build_evolution_circuit(ham_data, config) -> QuantumCircuit
    Build a SuzukiTrotter evolution circuit from a HamiltonianData.
"""
from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from ..config import EvolutionConfig
from ..loaders.hamlib_loader import HamiltonianData


def build_pauli_evolution_circuit(
    hamiltonian: SparsePauliOp,
    t: float = 1.0,
    synthesis_order: int = 2,
    synthesis_reps: int = 0,
) -> QuantumCircuit:
    """
    Build an evolution circuit using Qiskit's PauliEvolutionGate.

    Uses Qiskit's built-in ``SuzukiTrotter`` synthesis to decompose
    e^{-iHt}.  The circuit contains one PauliEvolutionGate; transpilation
    decomposes it into the target basis.

    Parameters
    ----------
    hamiltonian     : SparsePauliOp representing H.
    t               : total evolution time.
    synthesis_order : Suzuki-Trotter order (1 or 2; higher orders available).
    synthesis_reps  : number of synthesis repetitions.

    Returns
    -------
    QuantumCircuit containing one PauliEvolutionGate.
    """
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import SuzukiTrotter

    gate = PauliEvolutionGate(
        hamiltonian,
        time=t,
        synthesis=SuzukiTrotter(order=synthesis_order, reps=synthesis_reps),
    )
    qc = QuantumCircuit(hamiltonian.num_qubits)
    qc.append(gate, range(hamiltonian.num_qubits))

    while any(hasattr(inst.operation, "definition") and inst.operation.definition is not None
            for inst in qc.data):
                qc = qc.decompose()
    return qc


def build_evolution_circuit(
    ham_data: HamiltonianData,
    config: EvolutionConfig,
) -> QuantumCircuit:
    """
    Build a PauliEvolutionGate circuit for Hamiltonian time evolution.

    Parameters
    ----------
    ham_data : HamiltonianData  (from loaders.hamlib_loader.load_hamiltonian)
    config   : EvolutionConfig

    Returns
    -------
    QuantumCircuit containing one PauliEvolutionGate (decomposed by transpilation).
    """
    return build_pauli_evolution_circuit(
        ham_data.pauli_op,
        t=config.evolution_time,
        synthesis_order=config.synthesis_order,
        synthesis_reps=config.synthesis_reps,
    )
