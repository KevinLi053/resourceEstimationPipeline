"""
Top-level pipeline orchestration.

Executes the full workflow:

    HamLib HDF5
        → HamiltonianData          (loaders.hamlib_loader)
        → QuantumCircuit (raw)     (circuit.evolution)
        → QuantumCircuit (C+T)     (circuit.transpile)
        → EstimationResult (Azure) (estimators.azure)
        → EstimationResult (QT)    (estimators.qualtran)
        → ComparisonReport         (compare.metrics)

Both estimators receive the same canonical Clifford+T circuit.

Public API
----------
run(config) -> PipelineResult
    Execute the full pipeline and return all intermediate outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from qiskit import QuantumCircuit

from resourceEstimationPipeline.config import PipelineConfig
from resourceEstimationPipeline.loaders.hamlib_loader import HamiltonianData, load_hamiltonian
from resourceEstimationPipeline.circuit.evolution import build_evolution_circuit
from resourceEstimationPipeline.circuit.transpile import (
    circuit_stats,
    circuit_to_qasm,
    transpile_to_clifford_t,
)
from resourceEstimationPipeline.estimators.base import EstimationResult
from resourceEstimationPipeline.compare.metrics import ComparisonReport, compare


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """
    All intermediate and final outputs of a single pipeline run.

    Fields
    ------
    config          : PipelineConfig used for this run
    ham_data        : loaded Hamiltonian + metadata
    raw_circuit     : un-transpiled evolution circuit
    clifford_t_circuit : canonical Clifford+T circuit (shared by both estimators)
    raw_stats       : gate counts for the raw circuit
    ct_stats        : gate counts for the Clifford+T circuit
    qasm_source     : OpenQASM 3 export of the Clifford+T circuit
    azure_result    : EstimationResult from the Azure QDK estimator
    qualtran_result : EstimationResult from the Qualtran estimator
    comparison      : ComparisonReport (side-by-side metric comparison)
    errors          : mapping of step → error message for any failed step
    """

    config: PipelineConfig

    ham_data: Optional[HamiltonianData] = None
    raw_circuit: Optional[QuantumCircuit] = None
    clifford_t_circuit: Optional[QuantumCircuit] = None

    raw_stats: Optional[Dict] = None
    ct_stats: Optional[Dict] = None
    qasm_source: Optional[str] = None

    azure_result: Optional[EstimationResult] = None
    qualtran_result: Optional[EstimationResult] = None
    comparison: Optional[ComparisonReport] = None

    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        """True if at least one estimator produced a result."""
        return self.azure_result is not None or self.qualtran_result is not None

    def available_results(self) -> List[EstimationResult]:
        """Return all non-None EstimationResult objects."""
        out = []
        if self.azure_result is not None:
            out.append(self.azure_result)
        if self.qualtran_result is not None:
            out.append(self.qualtran_result)
        return out

    def print_summary(self) -> None:
        """Print a concise run summary to stdout."""
        from resourceEstimationPipeline.compare.tables import print_comparison

        print("=" * 70)
        print("PIPELINE SUMMARY")
        print("=" * 70)

        if self.ham_data:
            print(f"Hamiltonian : {self.ham_data.key}")
            print(f"  Qubits    : {self.ham_data.nqubits}")
            print(f"  Terms     : {self.ham_data.n_terms}")
            print(f"  one_norm  : {self.ham_data.one_norm}")

        if self.raw_stats:
            print(f"\nRaw circuit : depth={self.raw_stats['depth']}, "
                  f"gates={self.raw_stats['total_gates']}, "
                  f"qubits={self.raw_stats['num_qubits']}")

        if self.ct_stats:
            print(f"C+T circuit : depth={self.ct_stats['depth']}, "
                  f"gates={self.ct_stats['total_gates']}, "
                  f"T={self.ct_stats['t_count']}, "
                  f"Rz={self.ct_stats['rz_count']}, "
                  f"Clifford={self.ct_stats['clifford_count']}")

        if self.errors:
            print(f"\nErrors during run:")
            for step, msg in self.errors.items():
                print(f"  [{step}] {msg}")

        if self.comparison and self.succeeded:
            print()
            print_comparison(self.comparison)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run(
    config: Optional[PipelineConfig] = None,
    run_azure: bool = True,
    run_qualtran: bool = True,
) -> PipelineResult:
    """
    Execute the complete resource-estimation pipeline.

    Parameters
    ----------
    config       : PipelineConfig. If None, uses DEFAULT_CONFIG.
    run_azure    : bool  whether to run the Azure QDK estimator.
    run_qualtran : bool  whether to run the Qualtran estimator.

    Returns
    -------
    PipelineResult  with all intermediate outputs and EstimationResult objects.

    Steps
    -----
    1. Load Hamiltonian from HamLib HDF5.
    2. Build evolution circuit (method from config.evolution).
    3. Transpile to canonical Clifford+T basis.
    4. (Optionally) run Azure QDK estimator.
    5. (Optionally) run Qualtran estimator.
    6. Compare results.
    """
    if config is None:
        from resourceEstimationPipeline.config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG

    result = PipelineResult(config=config)

    # ── Step 1: Load Hamiltonian ──────────────────────────────────────────────
    try:
        result.ham_data = load_hamiltonian(config.hamlib)
        print(f"[1/6] Loaded Hamiltonian: {result.ham_data}")
    except Exception as exc:
        result.errors["load_hamiltonian"] = str(exc)
        print(f"[1/6] ERROR loading Hamiltonian: {exc}")
        return result

    # ── Step 2: Build evolution circuit ──────────────────────────────────────
    try:
        result.raw_circuit = build_evolution_circuit(result.ham_data, config.evolution)
        result.raw_stats = circuit_stats(result.raw_circuit.decompose())
        print(
            f"[2/6] Built PauliEvolutionGate circuit: "
            f"depth={result.raw_stats['depth']}, "
            f"gates={result.raw_stats['total_gates']}"
        )
    except Exception as exc:
        result.errors["build_circuit"] = str(exc)
        print(f"[2/6] ERROR building circuit: {exc}")
        return result

    # ── Step 3: Transpile to Clifford+T ──────────────────────────────────────
    try:
        result.clifford_t_circuit = transpile_to_clifford_t(
            result.raw_circuit, config.transpile
        )
        result.ct_stats = circuit_stats(result.clifford_t_circuit)
        result.qasm_source = circuit_to_qasm(result.clifford_t_circuit)
        print(
            f"[3/6] Transpiled to Clifford+T: "
            f"depth={result.ct_stats['depth']}, "
            f"T={result.ct_stats['t_count']}, "
            f"Rz={result.ct_stats['rz_count']}, "
            f"Clifford={result.ct_stats['clifford_count']}"
        )
    except Exception as exc:
        result.errors["transpile"] = str(exc)
        print(f"[3/6] ERROR transpiling: {exc}")
        return result

    # ── Step 4: Azure QDK estimator ───────────────────────────────────────────
    if run_azure:
        try:
            from resourceEstimationPipeline.estimators.azure import estimate as azure_estimate
            result.azure_result = azure_estimate(result.clifford_t_circuit, config)
            print(
                f"[4/6] Azure QDK: "
                f"phys_qubits={result.azure_result.physical_qubits:,}, "
                f"runtime={result.azure_result.runtime_seconds:.4f}s"
            )
        except Exception as exc:
            result.errors["azure_estimate"] = str(exc)
            print(f"[4/6] ERROR (Azure): {exc}")
    else:
        print("[4/6] Azure estimator skipped.")

    # ── Step 5: Qualtran estimator ────────────────────────────────────────────
    if run_qualtran:
        try:
            from resourceEstimationPipeline.estimators.qualtran import estimate as qt_estimate
            result.qualtran_result = qt_estimate(result.clifford_t_circuit, config)
            print(
                f"[5/6] Qualtran: "
                f"phys_qubits={result.qualtran_result.physical_qubits:,}, "
                f"runtime={result.qualtran_result.runtime_seconds:.4f}s"
            )
        except Exception as exc:
            result.errors["qualtran_estimate"] = str(exc)
            print(f"[5/6] ERROR (Qualtran): {exc}")
    else:
        print("[5/6] Qualtran estimator skipped.")

    # ── Step 6: Compare ───────────────────────────────────────────────────────
    available = result.available_results()
    if len(available) >= 1:
        try:
            result.comparison = compare(available)
            print(
                f"[6/6] Comparison: {len(result.comparison.shared_metrics)} shared metrics, "
                f"{len(result.comparison.differences)} numeric differences."
            )
        except Exception as exc:
            result.errors["compare"] = str(exc)
            print(f"[6/6] ERROR comparing: {exc}")
    else:
        print("[6/6] No results to compare.")

    return result
