"""
Metric-level comparison between EstimationResult objects.

Comparison code never depends on estimator internals — it operates only
on the fields of :class:`~estimators.base.EstimationResult`.

Public API
----------
compare(results) -> ComparisonReport
    Build a structured comparison from a list of EstimationResult objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..estimators.base import EstimationResult

import dataclasses
from ..circuit.transpile import circuit_stats

def enrich_from_circuit(result: EstimationResult, circuit) -> EstimationResult:
    """
    Fill None logical-gate fields in result using circuit_stats().
    Only overwrites fields that are currnetly None - never replaces
    a value the estimator itself provided.
    """
    stats = circuit_stats(circuit)
    gate_counts = stats["gate_counts"]

    overrides = {}
    enriched_rotation_count = result.rotation_count
    if result.rotation_count is None:
        enriched_rotation_count = stats["rz_count"]
        overrides["rotation_count"] = enriched_rotation_count

    if result.t_count is None:
        if result.t_per_rotation is not None and enriched_rotation_count:
            overrides["t_count"] = result.t_per_rotation * enriched_rotation_count
        else:
            overrides["t_count"] = stats["t_count"]

    if result.clifford_count is None:
        overrides["clifford_count"] = stats["clifford_count"]

    if result.toffoli_count is None:
        overrides["toffoli_count"] = gate_counts.get("ccx", 0)

    if result.measurement_count is None:
        overrides["measurement_count"] = gate_counts.get("measure", 0)

    if result.logical_depth is None:
        overrides["logical_depth"] = stats["depth"]

    if result.t_depth is None:
        overrides["t_depth"] = stats.get("t_depth")

    return dataclasses.replace(result, **overrides)


# ---------------------------------------------------------------------------
# Metric descriptors
# ---------------------------------------------------------------------------

# Ordered list of (key, label, description) tuples defining every comparable metric.
# Keys must match the keys returned by EstimationResult.as_dict().
# Metrics are grouped by concern; the order determines table row order.
#
# Metrics that genuinely cannot be compared (neither estimator exposes an equivalent):
#   - Qualtran logical_depth / t_depth: CompositeBloq has no time ordering.
#   - Azure rotation_synthesis_precision: Azure uses budget-fraction allocation,
#     not a per-rotation ε — the closest proxy is t_per_rotation.
#   - Azure cycle_time_us: Azure uses gate_time_ns + measurement_time_ns instead.
#   - Qualtran gate_time_ns / measurement_time_ns: Qualtran uses cycle_time_us.
#   - Azure physical_memory_qubits: exposed by Azure but not modelled by Qualtran.
METRIC_DESCRIPTORS: List[Tuple[str, str, str]] = [
    # ── Logical / algorithmic ─────────────────────────────────────────────────
    ("Logical qubits",
     "Logical qubits",
     "Algorithm logical qubit count"),
    ("Logical depth",
     "Logical depth",
     "Logical circuit depth (gate layers, not QEC rounds); not available from Qualtran"),
    ("Logical cycles",
     "Logical cycles",
     "Total QEC syndrome-measurement rounds for the algorithm"),

    # ── Gate counts ───────────────────────────────────────────────────────────
    ("T count",
     "T count",
     "Total T-gate count (including synthesised Rz gates)"),
    ("T depth",
     "T depth",
     "T-gate circuit depth (critical path); not available from Qualtran"),
    ("Clifford count",
     "Clifford count",
     "Total Clifford gate count"),
    ("Rotation count (arb. Rz)",
     "Rotation count",
     "Number of arbitrary-angle Rz rotations (before synthesis)"),
    ("Toffoli count",
     "Toffoli count",
     "Toffoli / CCZ gate count"),
    ("Measurement count",
     "Measurement count",
     "Number of measurements"),

    # ── Physical resources ────────────────────────────────────────────────────
    ("Physical qubits (total)",
     "Physical qubits (total)",
     "Total physical qubit count (compute + factory + memory)"),
    ("Physical compute qubits",
     "Physical compute qubits",
     "Physical qubits for the logical compute register"),
    ("Physical factory qubits",
     "Physical factory qubits",
     "Physical qubits for the magic-state factory"),
    ("Physical memory qubits",
     "Physical memory qubits",
     "Physical qubits for logical memory (Azure only; not modelled by Qualtran)"),

    # ── Timing ────────────────────────────────────────────────────────────────
    ("Runtime (s)",
     "Runtime (s)",
     "Estimated wall-clock runtime in seconds"),

    # ── Error & QEC ───────────────────────────────────────────────────────────
    ("Error budget",
     "Error budget",
     "Total error budget: Azure uses it directly; Qualtran reports it when global_error_budget is set in PipelineConfig"),
    ("Logical error rate",
     "Logical error rate",
     "Estimated total logical failure probability"),
    ("Code distance",
     "Code distance",
     "Surface-code data-block distance"),
    ("Logical cycle time (ns)",
     "Logical cycle time (ns)",
     "Time for one logical QEC cycle in ns (d syndrome-extraction rounds). "
     "Azure: from LATTICE_SURGERY instruction.time(1). "
     "Qualtran: derivable as code_distance × cycle_time_us × 1000."),
    ("Code cycle time (ns)",
     "Code cycle time (ns)",
     "Syndrome extraction cycle time in ns. "
     "Azure: CODE_CYCLE_TIME property on LATTICE_SURGERY instruction. "
     "Qualtran: approximately cycle_time_us × 1000 (not directly reported)."),

    # ── Factory ───────────────────────────────────────────────────────────────
    ("Factory type",
     "Factory type",
     "Magic-state factory model (Litinski19 / RoundBased / CCZ2T)"),
    ("Factory config",
     "Factory config",
     "Magic-state factory configuration string / description"),
    ("Number of factories",
     "Number of factories",
     "Parallel magic-state factories: Azure optimises this; Qualtran defaults to 1 (single factory, longer runtime)"),

    # ── Rotation synthesis ────────────────────────────────────────────────────
    ("T gates per rotation",
     "T gates per rotation",
     "T gates used to synthesise each arbitrary Rz"),
    ("Rotation synthesis precision (ε)",
     "Rotation synthesis precision (ε)",
     "Target synthesis error per Rz rotation (rz_eps); Qualtran only"),

    # ── Physical parameters (estimator assumptions) ───────────────────────────
    ("Physical error rate",
     "Physical error rate",
     "Physical gate error rate assumed by the estimator"),
    ("Cycle time (µs)",
     "Cycle time (µs)",
     "Surface-code cycle time in microseconds (Qualtran only)"),
    ("Gate time (ns)",
     "Gate time (ns)",
     "Single/two-qubit gate time in nanoseconds (Azure GateBased model only)"),
    ("Measurement time (ns)",
     "Measurement time (ns)",
     "Measurement time in nanoseconds (Azure GateBased model only)"),

    # ── Derived metrics ───────────────────────────────────────────────────────
    ("Physical qubits per logical qubit",
     "Physical qubits per logical qubit",
     "Total physical overhead per algorithmic qubit (physical_qubits / logical_qubits)"),
    ("Factory qubit fraction",
     "Factory qubit fraction",
     "Fraction of physical qubits devoted to the magic-state factory"),
    ("Runtime per T gate (s)",
     "Runtime per T gate (s)",
     "Amortised wall-clock time per T state consumed (runtime_seconds / t_count)"),
    ("T gates per logical qubit",
     "T gates per logical qubit",
     "Algorithm T-count density (t_count / logical_qubits)"),
    ("Logical cycles per T gate",
     "Logical cycles per T gate",
     "QEC rounds per T gate; approximate when T gates dominate the schedule"),
    ("Physical qubits per T state",
     "Physical qubits per T state",
     "Amortised factory qubits per T state (physical_factory_qubits / t_count)"),

    # ── Assumptions (text) ────────────────────────────────────────────────────
    ("Algorithm assumptions",
     "Algorithm assumptions",
     "Algorithmic assumptions made by the estimator"),
    ("Architecture assumptions",
     "Architecture assumptions",
     "Hardware / architecture assumptions"),
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
