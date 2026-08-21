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


def print_qubit_debug(results) -> None:
    """
    Print a side-by-side physical-qubit breakdown for debugging.

    Compares:
      Azure:    compute qubits | factory qubits | num_factories | total
      Qualtran: logical qubits | compute qubits | qubits_per_factory
                | num_factories | total factory qubits | total

    Also validates that compute + factory (+ memory) == total for each estimator.

    Parameters
    ----------
    results : ComparisonReport  OR  list of EstimationResult
    """
    from .metrics import ComparisonReport
    if isinstance(results, ComparisonReport):
        result_list = results.results
    else:
        result_list = list(results)

    sep = "─" * 72
    print(sep)
    print("  PHYSICAL QUBIT BREAKDOWN (DEBUG)")
    print(sep)

    for r in result_list:
        is_azure = "azure" in r.estimator_name.lower() or "qdk" in r.estimator_name.lower()
        print(f"\n  [{r.estimator_name}]")

        if is_azure:
            compute  = r.physical_compute_qubits
            factory  = r.physical_factory_qubits
            memory   = r.physical_memory_qubits
            n_fact   = r.num_factories
            total    = r.physical_qubits

            qpf      = (factory // n_fact) if (factory and n_fact) else None

            print(f"    Physical compute qubits      : {compute:>12,}" if compute is not None else "    Physical compute qubits      :          N/A")
            print(f"    Physical factory qubits (tot): {factory:>12,}" if factory is not None else "    Physical factory qubits (tot):          N/A")
            print(f"    Qubits per factory           : {qpf:>12,}"    if qpf     is not None else "    Qubits per factory           :          N/A")
            print(f"    Number of factories          : {n_fact:>12,}" if n_fact  is not None else "    Number of factories          :          N/A")
            print(f"    Physical memory qubits       : {memory:>12,}" if memory  is not None else "    Physical memory qubits       :          N/A")
            print(f"    Total physical qubits        : {total:>12,}"  if total   is not None else "    Total physical qubits        :          N/A")

            # Validation
            if compute is not None and factory is not None and total is not None:
                reconstructed = compute + factory + (memory or 0)
                match = "OK" if reconstructed == total else f"MISMATCH (got {reconstructed:,})"
                print(f"    Validation compute+factory+memory=total: {match}")

        else:
            # Qualtran
            logical  = r.logical_qubits
            compute  = r.physical_compute_qubits
            factory  = r.physical_factory_qubits
            n_fact   = r.num_factories
            total    = r.physical_qubits
            qpf      = r.extra.get("qubits_per_factory") if r.extra else None

            print(f"    Logical qubits               : {logical:>12,}" if logical is not None else "    Logical qubits               :          N/A")
            print(f"    Physical compute qubits      : {compute:>12,}" if compute is not None else "    Physical compute qubits      :          N/A")
            print(f"    Qubits per factory           : {qpf:>12,}"    if qpf     is not None else "    Qubits per factory           :          N/A")
            print(f"    Number of factories          : {n_fact:>12,}" if n_fact  is not None else "    Number of factories          :          N/A")
            print(f"    Total factory qubits         : {factory:>12,}" if factory is not None else "    Total factory qubits         :          N/A")
            print(f"    Total physical qubits        : {total:>12,}"  if total   is not None else "    Total physical qubits        :          N/A")

            if compute is not None and factory is not None and total is not None:
                reconstructed = compute + factory
                match = "OK" if reconstructed == total else f"MISMATCH (got {reconstructed:,})"
                print(f"    Validation compute+factory=total         : {match}")

        # Error budget info
        print(f"    Error budget                 : {r.error_budget}" if r.error_budget is not None else "    Error budget                 :          N/A")
        if r.rotation_synthesis_precision is not None:
            print(f"    Rotation synthesis precision : {r.rotation_synthesis_precision:.3e}")

    print()
    print(sep)
    print("  KEY MODEL DIFFERENCE:")
    print("  Azure uses N parallel factories (optimizer-chosen) to meet the error")
    print("  budget; total factory qubits = N × per-factory footprint.")
    print("  Qualtran (default n_factories=1) runs a single factory for more cycles;")
    print("  factory qubit count is therefore constant (independent of algorithm size).")
    print("  Set QualtranConfig.n_factories > 1 or use MultiFactory to match Azure.")
    print(sep)