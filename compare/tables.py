"""
Table rendering for comparison reports.

Converts :class:`~compare.metrics.ComparisonReport`
objects into pandas DataFrames and rich text output.

Public API
----------
comparison_dataframe(report)  -> pd.DataFrame
    Full side-by-side table (all metrics, N/A for missing).

differences_dataframe(report) -> pd.DataFrame
    Only metrics that differ between estimators.

print_comparison(report)      -> None
    Print a formatted comparison to stdout.

highlight_differences(report) -> None
    Print only the differing metrics.
"""
from __future__ import annotations

from typing import List, Optional

from .metrics import ComparisonReport, MetricComparison


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

def comparison_dataframe(report: ComparisonReport):
    """
    Build a full side-by-side comparison DataFrame.

    Every comparable metric is included; missing values are shown as 'N/A'.

    Parameters
    ----------
    report : ComparisonReport

    Returns
    -------
    pandas.DataFrame  with columns [Metric, <estimator 1>, <estimator 2>, ...]
    """
    import pandas as pd

    rows = []
    for mc in report.metric_rows:
        row = mc.as_display_dict()
        rows.append(row)

    df = pd.DataFrame(rows)
    # Rows where ratio is not applicable (non-numeric or one side missing) have no
    # "Ratio (B/A)" key in their dict, so pandas fills those cells with NaN.
    # Replace with "—" so the column stays all-string.
    if "Ratio (B/A)" in df.columns:
        df["Ratio (B/A)"] = df["Ratio (B/A)"].fillna("—")
    # Put Metric first
    cols = ["Metric"] + [c for c in df.columns if c != "Metric"]
    return df[cols]


def differences_dataframe(report: ComparisonReport):
    """
    Build a DataFrame showing only metrics that differ between estimators.

    Parameters
    ----------
    report : ComparisonReport

    Returns
    -------
    pandas.DataFrame  (subset of comparison_dataframe rows)
    """
    import pandas as pd

    rows = [mc.as_display_dict() for mc in report.differences]
    if not rows:
        return pd.DataFrame(columns=["Metric"] + report.estimator_names)
    df = pd.DataFrame(rows)
    cols = ["Metric"] + [c for c in df.columns if c != "Metric"]
    return df[cols]


def missing_dataframe(report: ComparisonReport):
    """
    Build a DataFrame showing which metrics are unavailable in which estimator.

    Parameters
    ----------
    report : ComparisonReport

    Returns
    -------
    pandas.DataFrame
    """
    import pandas as pd

    rows = []
    for mc in report.metric_rows:
        if not all(mc.available.values()):
            row = {"Metric": mc.metric}
            for name, present in mc.available.items():
                row[name] = "available" if present else "N/A"
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["Metric"] + report.estimator_names)
    df = pd.DataFrame(rows)
    cols = ["Metric"] + [c for c in df.columns if c != "Metric"]
    return df[cols]


# ---------------------------------------------------------------------------
# Text renderers
# ---------------------------------------------------------------------------

def _fmt(val, metric: str = "") -> str:
    """Format a metric value for fixed-width text output."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        if "error" in metric.lower() or "rate" in metric.lower() or "budget" in metric.lower():
            return f"{val:.3e}"
        return f"{val:.4g}"
    if isinstance(val, int):
        return f"{val:,}"
    return str(val)


def print_comparison(report: ComparisonReport, width: int = 100) -> None:
    """
    Print a full side-by-side comparison table to stdout.

    Parameters
    ----------
    report : ComparisonReport
    width  : int  total line width for formatting
    """
    names = report.estimator_names
    col_w = max(18, (width - 34) // max(len(names), 1))

    sep = "─" * width
    print(sep)
    print("  RESOURCE ESTIMATOR COMPARISON")
    print(sep)

    # Header
    header = f"  {'Metric':<32}"
    for n in names:
        header += f"  {n[:col_w]:<{col_w}}"
    print(header)
    print(sep)

    for mc in report.metric_rows:
        vals = mc.values
        row = f"  {mc.metric:<32}"
        for n in names:
            v = vals.get(n)
            row += f"  {_fmt(v, mc.metric):<{col_w}}"
        if mc.ratio is not None and abs(mc.ratio - 1.0) > 0.001:
            row += f"  ← {mc.ratio:.3f}× diff"
        print(row)

    print(sep)
    print(f"  Shared metrics   : {len(report.shared_metrics)}")
    print(f"  N/A in ≥1 est.  : {len(report.missing_metrics)}")
    print(f"  Key differences  : {len(report.differences)}")
    print(sep)


def highlight_differences(report: ComparisonReport) -> None:
    """
    Print only the metrics that differ between estimators.

    Parameters
    ----------
    report : ComparisonReport
    """
    names = report.estimator_names
    print("\n=== Metrics that differ between estimators ===")
    if not report.differences:
        print("  (no numeric differences found)")
        return

    for mc in report.differences:
        parts = []
        for n in names:
            v = mc.values.get(n)
            parts.append(f"{n}: {_fmt(v, mc.metric)}")
        ratio_str = f"  →  ratio {mc.ratio:.3f}×" if mc.ratio is not None else ""
        print(f"  {mc.metric}: {' | '.join(parts)}{ratio_str}")


def explain_differences(report: ComparisonReport) -> str:
    """
    Return a textual explanation of *why* the estimators produce different results.

    Focuses on structural differences between the two estimator models.

    Returns
    -------
    str  multi-paragraph explanation
    """
    lines = []
    lines.append("=" * 70)
    lines.append("WHY DO THE ESTIMATORS PRODUCE DIFFERENT RESULTS?")
    lines.append("=" * 70)
    lines.append("")
    lines.append(
        "Both estimators receive the SAME canonical Clifford+T circuit, so any "
        "differences arise purely from the resource estimation models, not the "
        "input circuit."
    )
    lines.append("")
    lines.append("Key model differences:")
    lines.append("")
    lines.append(
        "1. T-gate synthesis for arbitrary Rz rotations\n"
        "   • Azure QDK: reports NUM_TS_PER_ROTATION (the actual synthesis count\n"
        "     it uses internally for the given error budget).\n"
        "   • Qualtran: counts raw Rz gates from the bloq graph; T synthesis cost\n"
        "     is estimated post-hoc via the Solovay-Kitaev formula ~3·log₂(1/ε).\n"
        "   → Expect T-count differences when rotation_count > 0."
    )
    lines.append("")
    lines.append(
        "2. Error budget vs. code distance\n"
        "   • Azure QDK: takes max_error (total budget); sweeps code distance\n"
        "     internally to find the smallest code that satisfies the budget.\n"
        "   • Qualtran: takes code distance directly; logical error is computed\n"
        "     from the code, not the other way around.\n"
        "   → Comparing at the 'same' code distance is the closest apples-to-apples."
    )
    lines.append("")
    lines.append(
        "3. Magic-state factory models\n"
        "   • Azure QDK: Litinski-2019 or Round-based factory; independently tuned.\n"
        "   • Qualtran: CCZ2T factory from Gidney-Fowler 2019.\n"
        "   → Different factories produce different factory qubit counts."
    )
    lines.append("")
    lines.append(
        "4. Clifford gate cost\n"
        "   • Azure QDK: Clifford gates are essentially free in fault-tolerant\n"
        "     computation (they are absorbed into the syndrome schedule).\n"
        "   • Qualtran: counts Cliffords for completeness; they do not directly\n"
        "     increase runtime (T gates dominate).\n"
        "   → Clifford counts are informational only in both models."
    )
    lines.append("")
    lines.append(
        "5. Runtime model\n"
        "   • Azure QDK: time in ns based on gate_time + T-factory cycle time.\n"
        "   • Qualtran: time in hours based on CCZ-cycle counts × cycle_time_us.\n"
        "   → Runtime numbers are not directly comparable without unit conversion."
    )
    lines.append("")
    lines.append(
        "6. Physical qubit layout\n"
        "   • Azure QDK: qubit = code-distance² × 2 (standard surface-code overhead).\n"
        "   • Qualtran: SimpleDataBlock uses the same formula but the factory qubit\n"
        "     count depends on the CCZ2T factory parameters.\n"
        "   → Physical qubit counts should be close but not identical."
    )
    return "\n".join(lines)
