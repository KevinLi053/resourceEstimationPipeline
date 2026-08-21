"""
Multi-Hamiltonian comparison plots.

Creates a 6-row × 2-column figure comparing Azure and Qualtran estimates
across multiple HamLib Hamiltonians.

Left column x-axis  : Rz count
Right column x-axis : Logical qubits

Rows (top to bottom):
    Spacetime (qubit-seconds)
    Compute qubits
    Factory qubits
    Total qubits
    Runtime (s)
    T gate count

Public API
----------
plot_multi_ham_comparison(df, ...) -> matplotlib.figure.Figure
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# DataFrame column names (from multi_ham.COMPARISON_COLUMNS)
_ESTIMATOR_COL = "estimator"
_ESTIMATOR_NAMES = ("Azure", "Qualtran")  # canonical names after normalization

# Row definitions: (y_column, y_label)
_ROWS: List[Tuple[str, str]] = [
    ("spacetime",     "Spacetime\n(qubit·s)"),
    ("compute_qubits","Compute\nQubits"),
    ("factory_qubits","Factory\nQubits"),
    ("total_qubits",  "Total\nQubits"),
    ("runtime",       "Runtime (s)"),
    ("t_count",       "T Gate\nCount"),
]

# Column definitions: (x_column, x_label)
_COLS: List[Tuple[str, str]] = [
    ("rz_count",       "Rz Count"),
    ("logical_qubits", "Logical Qubits"),
]

# One colour per estimator — distinct and consistent across all 12 subplots
_COLORS: Dict[str, str] = {
    "Azure":    "#1f77b4",   # blue
    "Qualtran": "#d62728",   # red
}
_MARKERS: Dict[str, str] = {
    "Azure":    "o",
    "Qualtran": "s",
}


# ---------------------------------------------------------------------------
# Label helper
# ---------------------------------------------------------------------------

def _short_label(ham_key: str, nqubits: int | None = None) -> str:
    """Build a compact, readable label from a HamLib key.

    Examples
    --------
    'graph-1D-grid-nonpbc-qubitnodes_Lx-10_h-1'          → '1D grid\\nLx=10'
    'graph-2D-grid-pbc-qubitnodes_Lx-20_Ly-20_h-1'       → '2D grid\\n20×20'
    'graph-3D-grid-pbc-qubitnodes_Lx-10_Ly-10_Lz-10_h-1' → '3D grid\\n10×10×10'
    'fh-graph-2D-grid-nonpbc-qubitnodes_Lx-2_Ly-2_U-0...' → 'FH 2D grid\\n2×2'
    """
    k = ham_key
    fh = k.startswith("fh-")
    if fh:
        k = k[3:]

    m = re.match(r"graph-(\dD)-([^-]+)-[^-]+-qubitnodes_(.*)", k)
    if not m:
        return ham_key[-22:]

    dim, geom, params = m.group(1), m.group(2), m.group(3)
    sizes = re.findall(r"L[xyz]-(\d+)", params)
    size_str = "×".join(sizes) if sizes else params.split("_")[0]

    prefix = "FH " if fh else ""
    label = f"{prefix}{dim} {geom}\n{size_str}"
    if nqubits is not None:
        label += f" ({nqubits}q)"
    return label


# ---------------------------------------------------------------------------
# Core plotting function
# ---------------------------------------------------------------------------

def plot_multi_ham_comparison(
    df: pd.DataFrame,
    figsize: Tuple[float, float] = (14, 18),
    title: str = "Azure vs Qualtran — Multi-Hamiltonian Comparison",
    log_y: bool = True,
    annotate_points: bool = False,
) -> plt.Figure:
    """
    Build a 6-row × 2-column comparison figure.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame from run_multi_hamiltonian() or load_combined().
        Must have columns: estimator, rz_count, logical_qubits, spacetime,
        compute_qubits, factory_qubits, total_qubits, runtime, t_count.
    figsize : tuple
    title : str
    log_y : bool
        Use log scale on y-axes (recommended; many metrics span orders of
        magnitude).
    annotate_points : bool
        If True, label each point with a short ham_key identifier.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(
        nrows=len(_ROWS),
        ncols=len(_COLS),
        figsize=figsize,
        squeeze=False,
    )

    # Force numeric columns
    numeric_cols = [
        "rz_count", "logical_qubits", "spacetime",
        "compute_qubits", "factory_qubits", "total_qubits",
        "runtime", "t_count",
    ]
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Determine all estimator names present in data (keep canonical order)
    estimators_in_data = [e for e in _ESTIMATOR_NAMES if e in df[_ESTIMATOR_COL].unique()]

    for col_idx, (x_col, x_label) in enumerate(_COLS):
        for row_idx, (y_col, y_label) in enumerate(_ROWS):
            ax = axes[row_idx][col_idx]

            for est in estimators_in_data:
                colour = _COLORS.get(est, "grey")
                marker = _MARKERS.get(est, "o")

                sub = df[df[_ESTIMATOR_COL] == est][[x_col, y_col, "ham_key"]].copy()
                sub = sub.dropna(subset=[x_col, y_col])
                if sub.empty:
                    continue

                # Sort by x so connecting line is deterministic
                sub = sub.sort_values(x_col).reset_index(drop=True)

                ax.plot(
                    sub[x_col],
                    sub[y_col],
                    color=colour,
                    marker=marker,
                    linewidth=1.4,
                    markersize=6,
                    label=est,
                )

                if annotate_points and "ham_key" in sub.columns:
                    nq_col = "nqubits" if "nqubits" in sub.columns else None
                    for idx, (_, row) in enumerate(sub.iterrows()):
                        nq = int(row[nq_col]) if nq_col else None
                        label = _short_label(str(row["ham_key"]), nq)
                        # Alternate label offset above/below to reduce overlap
                        y_off = 6 if idx % 2 == 0 else -18
                        ax.annotate(
                            label,
                            (row[x_col], row[y_col]),
                            fontsize=5.5,
                            xytext=(4, y_off),
                            textcoords="offset points",
                            color=colour,
                            alpha=0.85,
                            bbox=dict(
                                boxstyle="round,pad=0.15",
                                fc="white",
                                ec="none",
                                alpha=0.6,
                            ),
                        )

            # Axes styling
            if log_y:
                ax.set_yscale("log")

            # y-label only on left column
            if col_idx == 0:
                ax.set_ylabel(y_label, fontsize=9)

            # x-label only on bottom row
            if row_idx == len(_ROWS) - 1:
                ax.set_xlabel(x_label, fontsize=9)

            # Column header on top row
            if row_idx == 0:
                ax.set_title(x_label, fontsize=10, fontweight="bold", pad=6)

            ax.tick_params(axis="both", labelsize=7)
            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)

    # Shared legend below the top-right subplot
    legend_handles = [
        mlines.Line2D(
            [], [],
            color=_COLORS.get(e, "grey"),
            marker=_MARKERS.get(e, "o"),
            markersize=6,
            linewidth=1.4,
            label=e,
        )
        for e in estimators_in_data
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(estimators_in_data),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 1.0),
    )

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig
