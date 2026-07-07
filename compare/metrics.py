"""
Metric-level comparison between EstimationResult objects.

Comparison code never depends on estimator internals — it operates only
on the fields of :class:`~resourceEstimationPipeline.estimators.base.EstimationResult`.

Public API
----------
compare(results) -> ComparisonReport
    Build a structured comparison from a list of EstimationResult objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from resourceEstimationPipeline.estimators.base import EstimationResult


# ---------------------------------------------------------------------------
# Metric descriptors
# ---------------------------------------------------------------------------

# Ordered list of (key, label, description) tuples defining every comparable metric.
# Keys match the field names in EstimationResult.as_dict().
METRIC_DESCRIPTORS: List[Tuple[str, str, str]] = [
    ("Logical qubits",           "Logical qubits",             "Algorithm logical qubit count"),
    ("Physical qubits (total)",  "Physical qubits (total)",    "Total physical qubit count (compute + factory + memory)"),
    ("Physical compute qubits",  "Physical compute qubits",    "Physical qubits for the logical compute register"),
    ("Physical factory qubits",  "Physical factory qubits",    "Physical qubits for the magic-state factory"),
    ("Physical memory qubits",   "Physical memory qubits",     "Physical qubits for logical memory"),
    ("T count",                  "T count",                    "Total T-gate count (including synthesised Rz gates)"),
    ("T depth",                  "T depth",                    "T-gate circuit depth (T gates on the critical path)"),
    ("Clifford count",           "Clifford count",             "Total Clifford gate count"),
    ("Rotation count (arb. Rz)", "Rotation count",             "Number of arbitrary-angle Rz rotations"),
    ("Toffoli count",            "Toffoli count",              "Toffoli / CCZ gate count"),
    ("Measurement count",        "Measurement count",          "Number of measurements"),
    ("Runtime (s)",              "Runtime (s)",                "Estimated wall-clock runtime in seconds"),
    ("Error budget",             "Error budget",               "Total error budget supplied to the estimator"),
    ("Logical error rate",       "Logical error rate",         "Estimated total logical failure probability"),
    ("Code distance",            "Code distance",              "Surface-code data-block distance"),
    ("Factory config",           "Factory config",             "Magic-state factory configuration string"),
    ("T gates per rotation",     "T gates per rotation",       "T gates used to synthesise each arbitrary Rz"),
    ("Algorithm assumptions",    "Algorithm assumptions",      "Algorithmic assumptions made by the estimator"),
    ("Architecture assumptions", "Architecture assumptions",   "Hardware / architecture assumptions"),
]


# ---------------------------------------------------------------------------
# Comparison entry
# ---------------------------------------------------------------------------

@dataclass
class MetricComparison:
    """Comparison of a single metric across all estimators."""

    metric: str
    description: str
    values: Dict[str, Any]      # {estimator_name: value_or_None}
    available: Dict[str, bool]  # {estimator_name: True/False}
    ratio: Optional[float] = None  # value[1] / value[0] if both numeric, else None

    def as_display_dict(self) -> Dict[str, str]:
        """Return values formatted for table display ('N/A' for missing)."""
        out: Dict[str, str] = {"Metric": self.metric}
        for name, val in self.values.items():
            if val is None:
                out[name] = "N/A"
            elif isinstance(val, float):
                out[name] = f"{val:.4g}"
            elif isinstance(val, int):
                out[name] = f"{val:,}"
            else:
                out[name] = str(val)
        if self.ratio is not None:
            out["Ratio (B/A)"] = f"{self.ratio:.3f}×"
        return out


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

@dataclass
class ComparisonReport:
    """
    Full side-by-side comparison between two or more EstimationResult objects.

    Attributes
    ----------
    results         : list of EstimationResult objects
    metric_rows     : one MetricComparison per metric
    differences     : metrics where values differ between estimators (numeric only)
    shared_metrics  : metrics available in all estimators
    missing_metrics : metrics not available in at least one estimator
    """

    results: List[EstimationResult]
    metric_rows: List[MetricComparison]
    differences: List[MetricComparison] = field(default_factory=list)
    shared_metrics: List[str] = field(default_factory=list)
    missing_metrics: List[str] = field(default_factory=list)

    @property
    def estimator_names(self) -> List[str]:
        return [r.estimator_name for r in self.results]

    def get_metric(self, metric: str) -> Optional[MetricComparison]:
        """Look up a MetricComparison by metric name."""
        for m in self.metric_rows:
            if m.metric == metric:
                return m
        return None

    def summary_text(self) -> str:
        """Return a human-readable comparison summary."""
        lines = ["=== Resource Estimator Comparison ===", ""]
        for name in self.estimator_names:
            lines.append(f"  {name}")
        lines.append("")
        lines.append(f"Shared metrics    : {len(self.shared_metrics)}")
        lines.append(f"Missing (≥1 est.) : {len(self.missing_metrics)}")
        lines.append("")
        lines.append("--- Key differences (numeric metrics) ---")
        for m in self.differences:
            vals = list(m.values.values())
            ratio_str = f"  ratio={m.ratio:.3f}×" if m.ratio is not None else ""
            lines.append(f"  {m.metric}: {vals}{ratio_str}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def compare(results: List[EstimationResult]) -> ComparisonReport:
    """
    Build a :class:`ComparisonReport` from a list of EstimationResult objects.

    Comparison code never depends on estimator internals — it operates only
    on the public fields returned by ``result.as_dict()``.

    Parameters
    ----------
    results : list of EstimationResult  (typically one Azure, one Qualtran)

    Returns
    -------
    ComparisonReport
    """
    if not results:
        raise ValueError("At least one EstimationResult is required.")

    # Build one dict per result
    dicts = [r.as_dict() for r in results]
    names = [r.estimator_name for r in results]

    metric_rows: List[MetricComparison] = []
    shared: List[str] = []
    missing: List[str] = []
    differences: List[MetricComparison] = []

    for key, label, description in METRIC_DESCRIPTORS:
        values: Dict[str, Any] = {}
        available: Dict[str, bool] = {}

        for name, d in zip(names, dicts):
            v = d.get(key)
            values[name] = v
            available[name] = v is not None

        all_present = all(available.values())
        none_present = not any(available.values())

        # Compute ratio for numeric pairs
        ratio: Optional[float] = None
        if len(results) == 2 and all_present:
            v0, v1 = list(values.values())
            if isinstance(v0, (int, float)) and isinstance(v1, (int, float)) and v0 != 0:
                try:
                    ratio = float(v1) / float(v0)
                except (ZeroDivisionError, TypeError):
                    pass

        mc = MetricComparison(
            metric=label,
            description=description,
            values=values,
            available=available,
            ratio=ratio,
        )
        metric_rows.append(mc)

        if all_present:
            shared.append(label)
        else:
            missing.append(label)

        # Flag numeric differences
        if all_present and len(results) == 2 and ratio is not None:
            if abs(ratio - 1.0) > 0.001:  # more than 0.1% difference
                differences.append(mc)

    return ComparisonReport(
        results=results,
        metric_rows=metric_rows,
        differences=differences,
        shared_metrics=shared,
        missing_metrics=missing,
    )
