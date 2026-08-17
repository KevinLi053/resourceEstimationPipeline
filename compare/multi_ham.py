"""
Multi-Hamiltonian pipeline runner and result persistence.

Runs Azure+Qualtran estimation for a list of HamLib Hamiltonians and
collects results into a unified tidy DataFrame (one row per estimator
per Hamiltonian), grouped by problem type.

Public API
----------
run_multi_hamiltonian(ham_specs, base_cfg, ...) -> pd.DataFrame
save_comparison(df, ham_key, group, out_dir)   -> Path
load_comparison(ham_key, group, out_dir)       -> pd.DataFrame | None
save_sidebyside(df, ham_key, group, out_dir)   -> Path
save_combined(df, out_dir)                     -> Path
load_combined(out_dir)                         -> pd.DataFrame | None
"""
from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ..estimators.base import EstimationResult
from ..loaders.hamlib_loader import load_hamiltonian as _load_hamiltonian
from .. import pipeline as _pipeline

# ---------------------------------------------------------------------------
# DataFrame schema
# ---------------------------------------------------------------------------

COMPARISON_COLUMNS = [
    "group",
    "ham_key",
    "hdf5_path",
    "key_index",
    "nqubits",
    "estimator",       # "Azure" or "Qualtran"
    "rz_count",        # circuit Rz count (x-axis 1)
    "logical_qubits",  # logical qubit count from estimator (x-axis 2)
    "spacetime",       # total_qubits * runtime  [qubit-seconds]
    "compute_qubits",
    "factory_qubits",
    "total_qubits",
    "runtime",         # seconds
    "t_count",
    "estimator_name",  # full string for debugging
]


def _detect_group(hdf5_path: str) -> str:
    """Extract problem type from the HDF5 file path.

    Matches known condensed-matter and optimization category names found in
    the HamLib directory tree.  Falls back to the HDF5 file stem.
    """
    known = [
        "heisenberg", "fermihubbard", "bosehubbard", "hubbard",
        "maxkcut", "lattices", "chemistry", "discreteoptimization",
        "ising", "tfim", "xxz",
    ]
    parts = [p.lower() for p in Path(hdf5_path).parts]
    for part in reversed(parts):
        for name in known:
            if name in part:
                return name
    return Path(hdf5_path).stem.lower()


def normalize_estimator(name: str) -> str:
    nl = name.lower()
    if "azure" in nl or "qdk" in nl:
        return "Azure"
    if "qualtran" in nl:
        return "Qualtran"
    return name


def _result_to_row(
    group: str,
    ham_key: str,
    hdf5_path: str,
    key_index: int,
    nqubits: int,
    rz_count: int,
    result: EstimationResult,
) -> Dict[str, Any]:
    pq = result.physical_qubits
    rt = result.runtime_seconds
    spacetime = (pq * rt) if (pq is not None and rt is not None) else None
    return {
        "group": group,
        "ham_key": ham_key,
        "hdf5_path": str(hdf5_path),
        "key_index": int(key_index),
        "nqubits": int(nqubits),
        "estimator": normalize_estimator(result.estimator_name),
        "rz_count": int(rz_count),
        "logical_qubits": result.logical_qubits,
        "spacetime": spacetime,
        "compute_qubits": result.physical_compute_qubits,
        "factory_qubits": result.physical_factory_qubits,
        "total_qubits": pq,
        "runtime": rt,
        "t_count": result.t_count,
        "estimator_name": result.estimator_name,
    }


def _sanitize_key(ham_key: str) -> str:
    return re.sub(r"[/\\:*?\"<>|]", "_", ham_key)[:200]


# ---------------------------------------------------------------------------
# Per-Hamiltonian summary save / load  (used for skip_cached)
# ---------------------------------------------------------------------------

def save_comparison(
    df: pd.DataFrame,
    ham_key: str,
    group: str,
    out_dir: Path,
) -> Path:
    """Save per-Hamiltonian summary rows (flattened) to out_dir/group/."""
    d = Path(out_dir) / group
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"summary_{_sanitize_key(ham_key)}.csv"
    df.to_csv(path, index=False)
    return path


def load_comparison(
    ham_key: str,
    group: str,
    out_dir: Path,
) -> Optional[pd.DataFrame]:
    """Return saved per-Hamiltonian summary CSV, or None if not found."""
    path = Path(out_dir) / group / f"summary_{_sanitize_key(ham_key)}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Per-Hamiltonian side-by-side comparison table  (like comparison.ipynb)
# ---------------------------------------------------------------------------

def save_sidebyside(
    df: pd.DataFrame,
    ham_key: str,
    group: str,
    out_dir: Path,
) -> Path:
    """Save the full side-by-side metric table (Metric | Azure | Qualtran | Ratio)."""
    d = Path(out_dir) / group
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"comparison_{_sanitize_key(ham_key)}.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Combined save / load
# ---------------------------------------------------------------------------

def save_combined(df: pd.DataFrame, out_dir: Path) -> Path:
    """Save combined all-Hamiltonians summary DataFrame to out_dir/."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "all_comparisons.csv"
    df.to_csv(path, index=False)
    return path


def load_combined(out_dir: Path) -> Optional[pd.DataFrame]:
    """Return the combined summary CSV, or None if not found."""
    path = Path(out_dir) / "all_comparisons.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Multi-Hamiltonian runner
# ---------------------------------------------------------------------------

def run_multi_hamiltonian(
    ham_specs: List[Dict],
    base_cfg,
    run_azure: bool = True,
    run_qualtran: bool = True,
    out_dir: Optional[Path] = None,
    skip_cached: bool = True,
    qualtran_estimator_fn: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Run Azure+Qualtran estimation for multiple HamLib Hamiltonians.

    Parameters
    ----------
    ham_specs : list of dicts, each containing:
        - hdf5_path (str, required)
        - key_index (int) or key (str) — at least one required
        - group (str, optional) — problem type; auto-detected from hdf5_path
          if omitted (e.g. "heisenberg", "fermihubbard", "bosehubbard")
    base_cfg : PipelineConfig
    run_azure, run_qualtran : bool
    out_dir : Path
        Root output directory. Per-group results go under out_dir/<group>/.
        - out_dir/<group>/summary_<key>.csv     flattened rows (skip_cached key)
        - out_dir/<group>/comparison_<key>.csv  side-by-side table per Hamiltonian
        - out_dir/all_comparisons.csv           combined summary
    skip_cached : bool
        Skip estimation when summary_<key>.csv already exists.
    qualtran_estimator_fn : callable, optional
        Custom Qualtran estimator with signature ``fn(circuit, config) -> EstimationResult``.
        When provided, the standard ``qualtran.estimate`` is bypassed and this
        function is called instead.  ``run_qualtran`` must still be True.
        Defaults to None (uses the standard pipeline estimator).

    Returns
    -------
    pd.DataFrame  (one row per estimator per Hamiltonian; includes 'group' column)
    """
    from ..config import HamlibConfig, PipelineConfig
    from ..compare.metrics import compare as _compare, enrich_from_circuit
    from ..compare.tables import comparison_dataframe as _comparison_dataframe

    if out_dir is None:
        raise ValueError("out_dir is required — pass the experiment output directory.")
    out_dir = Path(out_dir)

    all_rows: List[Dict] = []

    for i, spec in enumerate(ham_specs):
        hdf5_path = spec["hdf5_path"]
        key_index = spec.get("key_index")
        key = spec.get("key")
        group = spec.get("group") or _detect_group(hdf5_path)

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(ham_specs)}] [{group}]  idx={key_index}  key={key!r}")
        print(f"{'='*60}")

        ham_cfg = HamlibConfig(
            hdf5_path=hdf5_path,
            key=key,
            key_index=key_index if key_index is not None else 0,
        )
        cfg = PipelineConfig(
            hamlib=ham_cfg,
            evolution=base_cfg.evolution,
            transpile=base_cfg.transpile,
            azure=base_cfg.azure,
            qualtran=base_cfg.qualtran,
        )

        # Lightweight HamLib load — HDF5 key lookup only, no QASM/circuit
        try:
            ham_data = _load_hamiltonian(ham_cfg)
        except Exception as exc:
            print(f"  ERROR loading Hamiltonian: {exc}")
            continue

        if ham_data is None:
            print(f"  ERROR: load_hamiltonian returned None — skipping.")
            continue

        ham_key = ham_data.key
        nqubits = ham_data.nqubits

        # Check per-Hamiltonian cache before doing any circuit work
        if skip_cached:
            cached = load_comparison(ham_key, group, out_dir)
            if cached is not None:
                print(f"  Cache hit — loaded '{ham_key}'")
                all_rows.extend(cached.to_dict(orient="records"))
                continue

        # Run estimation (loads circuit + runs estimators)
        try:
            _run_qt_in_pipeline = run_qualtran and (qualtran_estimator_fn is None)
            pr = _pipeline.run(cfg, run_azure=run_azure, run_qualtran=_run_qt_in_pipeline)
        except Exception as exc:
            print(f"  ERROR (estimation): {exc}")
            continue

        # Inject custom Qualtran estimator result if provided
        if run_qualtran and qualtran_estimator_fn is not None:
            if pr.clifford_t_circuit is not None:
                try:
                    pr.qualtran_result = qualtran_estimator_fn(pr.clifford_t_circuit, cfg)
                    print(
                        f"[5/6] Qualtran (custom): "
                        f"phys_qubits={pr.qualtran_result.physical_qubits:,}, "
                        f"runtime={pr.qualtran_result.runtime_seconds:.4f}s"
                    )
                except Exception as exc:
                    pr.errors["qualtran_estimate"] = str(exc)
                    print(f"[5/6] ERROR (Qualtran custom): {exc}")
            else:
                print("[5/6] Qualtran (custom) skipped — no circuit available.")

        rz_count = (pr.ct_stats or {}).get("rz_count", 0)

        if pr.clifford_t_circuit is not None:
            if pr.azure_result is not None:
                pr.azure_result = enrich_from_circuit(
                    pr.azure_result, pr.clifford_t_circuit
                )
            if pr.qualtran_result is not None:
                pr.qualtran_result = enrich_from_circuit(
                    pr.qualtran_result, pr.clifford_t_circuit
                )

        available = pr.available_results()
        if not available:
            print(f"  WARNING: no results for '{ham_key}'")
            for step, msg in pr.errors.items():
                print(f"    [{step}]: {msg}")
            del pr
            gc.collect()
            continue

        # Save side-by-side comparison table (Metric | Azure | Qualtran | Ratio)
        try:
            report = _compare(available)
            sidebyside_df = _comparison_dataframe(report)
            sb_path = save_sidebyside(sidebyside_df, ham_key, group, out_dir)
            print(f"  Side-by-side → {sb_path}")
        except Exception as exc:
            print(f"  WARNING: could not build comparison table: {exc}")

        # Save flattened summary rows (for caching and plotting)
        ham_rows = [
            _result_to_row(group, ham_key, hdf5_path, key_index or 0, nqubits, rz_count, r)
            for r in available
        ]
        ham_df = pd.DataFrame(ham_rows, columns=COMPARISON_COLUMNS)
        summary_path = save_comparison(ham_df, ham_key, group, out_dir)
        print(f"  Summary      → {summary_path}")

        all_rows.extend(ham_rows)

        # Free the large QuantumCircuit before processing the next Hamiltonian
        del pr
        gc.collect()

    combined_df = (
        pd.DataFrame(all_rows, columns=COMPARISON_COLUMNS)
        if all_rows
        else pd.DataFrame(columns=COMPARISON_COLUMNS)
    )
    return combined_df
