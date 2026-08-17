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

import logging
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

from qiskit import QuantumCircuit

from resourceEstimationPipeline.config import PipelineConfig
from resourceEstimationPipeline.loaders.hamlib_loader import HamiltonianData, load_hamiltonian
from resourceEstimationPipeline.circuit.evolution import build_evolution_circuit
from resourceEstimationPipeline.circuit.transpile import (
    circuit_stats,
    circuit_to_qasm,
    transpile_to_clifford_t,
)
from resourceEstimationPipeline.circuit.cache import (
    evolved_path as _evolved_cache_path,
    final_circuit_path as _final_circuit_cache_path,
    save_circuit as _cache_save,
    load_circuit as _cache_load,
    resolve_index as _resolve_key_index,
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
    use_cache: bool = True,
) -> PipelineResult:
    """
    Execute the complete resource-estimation pipeline.

    Parameters
    ----------
    config       : PipelineConfig. If None, uses DEFAULT_CONFIG.
    run_azure    : bool  whether to run the Azure QDK estimator.
    run_qualtran : bool  whether to run the Qualtran estimator.
    use_cache    : bool  when False, always build and transpile fresh — never
                         reads from or writes to the QASM circuit cache.

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

    # ── Steps 2–3: Build / load evolution circuit, then synthesise to C+T ────
    #
    # Three-level cache:
    #   1. bqskit_<idx>.qasm exists → load and skip both build and synthesis
    #   2. evolved_<idx>.qasm exists → load raw circuit, run BQSKit, save result
    #   3. Neither exists → build circuit, save evolved, run BQSKit, save result
    try:
        _idx = _resolve_key_index(config.hamlib)
        _hdf5 = str(config.hamlib.hdf5_path)
        _ev_path = _evolved_cache_path(_hdf5, _idx)

        # Determine synthesis mode for cache key — must match transpile config.
        _synth_enabled = (
            config.transpile.rotation_synthesis_enabled
            and config.transpile.synthesis_strategy != "passthrough"
        )
        if not _synth_enabled:
            _synth_mode = "passthrough"
        else:
            _synth_mode = config.transpile.synthesis_method.lower().strip()
            if _synth_mode not in ("bqskit", "solovay_kitaev"):
                _synth_mode = "bqskit"  # safe fallback

        _final_path = _final_circuit_cache_path(_hdf5, _idx, _synth_mode)
    except Exception as exc:
        result.errors["cache_init"] = str(exc)
        print(f"[2/6] ERROR resolving cache paths: {exc}")
        return result

    _mode_label = (
        "Clifford+T" if _synth_enabled
        else "passthrough (rotations preserved)"
    )

    if use_cache and _final_path.exists():
        # ── Fast path: final circuit already cached ───────────────────────────
        print(f"[2/6] Cache hit ({_synth_mode}) — skipping evolution and synthesis.")
        try:
            result.clifford_t_circuit = _cache_load(_final_path)
            result.raw_stats = {}   # raw circuit not loaded in fast path
            result.ct_stats = circuit_stats(result.clifford_t_circuit)
            result.qasm_source = circuit_to_qasm(result.clifford_t_circuit)
            print(
                f"[3/6] (skipped — loaded from {_synth_mode} cache): "
                f"depth={result.ct_stats['depth']}, "
                f"T={result.ct_stats['t_count']}, "
                f"Rz={result.ct_stats['rz_count']}, "
                f"Clifford={result.ct_stats['clifford_count']}"
            )
        except Exception as exc:
            result.errors["cache_load_final"] = str(exc)
            print(f"[2/6] ERROR loading final circuit cache: {exc}")
            return result

    else:
        # ── Step 2: Build or load evolved circuit ─────────────────────────────
        if use_cache and _ev_path.exists():
            print(f"[2/6] Evolved cache hit — loading pre-built raw circuit.")
            try:
                result.raw_circuit = _cache_load(_ev_path)
                result.raw_stats = circuit_stats(result.raw_circuit)
                print(
                    f"[2/6] Loaded evolved circuit: "
                    f"depth={result.raw_stats['depth']}, "
                    f"gates={result.raw_stats['total_gates']}"
                )
            except Exception as exc:
                result.errors["cache_load_evolved"] = str(exc)
                print(f"[2/6] ERROR loading evolved cache: {exc}")
                return result
        else:
            try:
                result.raw_circuit = build_evolution_circuit(result.ham_data, config.evolution)
                result.raw_stats = circuit_stats(result.raw_circuit)
                print(
                    f"[2/6] Built PauliEvolutionGate circuit: "
                    f"depth={result.raw_stats['depth']}, "
                    f"gates={result.raw_stats['total_gates']}"
                )
                if use_cache:
                    _cache_save(result.raw_circuit, _ev_path)
                    print(f"[2/6] Saved evolved circuit → {_ev_path}")
            except Exception as exc:
                result.errors["build_circuit"] = str(exc)
                print(f"[2/6] ERROR building circuit: {exc}")
                return result

        # ── Step 3: Transpile / synthesize ───────────────────────────────────
        try:
            result.clifford_t_circuit = transpile_to_clifford_t(
                result.raw_circuit, config.transpile
            )
            result.ct_stats = circuit_stats(result.clifford_t_circuit)
            result.qasm_source = circuit_to_qasm(result.clifford_t_circuit)
            print(
                f"[3/6] Transpiled ({_mode_label}): "
                f"depth={result.ct_stats['depth']}, "
                f"T={result.ct_stats['t_count']}, "
                f"Rz={result.ct_stats['rz_count']}, "
                f"Clifford={result.ct_stats['clifford_count']}"
            )
            if use_cache:
                _cache_save(result.clifford_t_circuit, _final_path)
                print(f"[3/6] Saved {_synth_mode} circuit → {_final_path}")
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
            # ── Inject Azure parameters into Qualtran config (optional) ─────
            # When cfg.qualtran.use_azure_parameters=True and azure_result is
            # available, override Qualtran's QEC parameters with the values
            # Azure chose.  This enables "Mode 1: Azure-matched estimation"
            # where Qualtran estimates resources using Azure's fixed distance
            # and factory count instead of its own sweep/defaults.
            azure_params = None
            if run_azure and result.azure_result is not None:
                from resourceEstimationPipeline.estimators.azure import (
                    extract_azure_parameters as _extract_azure_params,
                )
                azure_params = _extract_azure_params(result.azure_result)
                use_azure = config.qualtran.use_azure_parameters
                d = azure_params.get("code_distance")
                n = azure_params.get("num_factories")
                if use_azure and (d is not None or n is not None):
                    # Build a shallow copy of QualtranConfig with Azure overrides.
                    # This leaves the original config object untouched so other
                    # pipeline runs remain unaffected.
                    from resourceEstimationPipeline.config import QualtranConfig

                    az_overrides: dict = {}
                    if d is not None:
                        az_overrides["data_d_sweep"] = [d]  # sweep single Azure distance
                    if n is not None:
                        az_overrides["n_factories"] = n
                    if not config.qualtran.data_d_sweep:
                        az_overrides["use_azure_parameters"] = True

                    config = PipelineConfig(
                        hamlib=config.hamlib,
                        evolution=config.evolution,
                        transpile=config.transpile,
                        azure=config.azure,
                        qualtran=QualtranConfig(**{**asdict(config.qualtran), **az_overrides}),
                    )
                    log.debug(
                        "Azure → Qualtran injected: d=%s  n_factories=%s",
                        d, n,
                    )

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
