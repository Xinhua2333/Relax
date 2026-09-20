# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from typing import Any, Iterable

from relax.utils.speculative import SpeculativeCounts, SpeculativeGeneration


def _counter_metrics(
    counts: list[SpeculativeCounts],
    total: int,
    prefix: str,
    *,
    coverage_known: bool = True,
) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {}
    for numerator, denominator, ratio_name, cohort in (
        ("accepted", "proposed", "accept_rate", "accept"),
        ("completion", "verify", "tokens_per_verify", "verify"),
    ):
        pairs = [(getattr(item, numerator), getattr(item, denominator)) for item in counts]
        pairs = [(top, bottom) for top, bottom in pairs if top is not None and bottom is not None]
        top_sum = sum(top for top, _ in pairs)
        bottom_sum = sum(bottom for _, bottom in pairs)
        metrics[f"{prefix}{numerator}_total"] = top_sum
        metrics[f"{prefix}{denominator}_total"] = bottom_sum
        metrics[f"{prefix}{cohort}_covered_count"] = len(pairs)
        metrics[f"{prefix}{cohort}_uncovered_count"] = total - len(pairs)
        if total and coverage_known:
            metrics[f"{prefix}{cohort}_count_coverage"] = len(pairs) / total
        if bottom_sum > 0:
            metrics[f"{prefix}{ratio_name}"] = top_sum / bottom_sum
    return metrics


def compute_speculative_metrics(samples: Iterable[Any]) -> dict[str, int | float]:
    """Reduce one exported batch; no identity survives into the next call."""
    generations: dict[tuple[str, str], SpeculativeGeneration] = {}
    conflicts: set[tuple[str, str]] = set()
    ordinary_counts: list[SpeculativeCounts] = []
    agentic_samples = legacy_samples = invalid_records = record_occurrences = 0
    for sample in samples:
        records = getattr(sample, "spec_generations", None)
        if records is None:
            info = getattr(sample, "spec_info", None)
            counts = getattr(info, "counts", None)
            # Agentic trace without records is old data, not an ordinary sample.
            metadata = getattr(sample, "metadata", None) or {}
            if counts is not None and "agentic_trace" not in metadata:
                ordinary_counts.append(counts)
            else:
                legacy_samples += 1
            continue
        agentic_samples += 1
        if not isinstance(records, list):
            invalid_records += 1
            continue
        for raw_record in records:
            record_occurrences += 1
            record = SpeculativeGeneration.from_dict(raw_record)
            sample_session = getattr(sample, "session_id", None)
            if record is None or (sample_session is not None and sample_session != record.session_id):
                invalid_records += 1
                continue
            key = (record.session_id, record.generation_id)
            existing = generations.get(key)
            if existing is not None and existing != record:
                conflicts.add(key)
            generations[key] = record

    metrics: dict[str, int | float] = {
        "spec/agentic_sample_count": agentic_samples,
        "spec/ordinary_sample_count": len(ordinary_counts),
        "spec/legacy_sample_count": legacy_samples,
        "spec/record_occurrence_count": record_occurrences,
        "spec/unique_generation_count": len(generations),
        "spec/conflicting_generation_count": len(conflicts),
        "spec/invalid_record_count": invalid_records,
    }
    metrics.update(
        _counter_metrics(
            [record.counts for key, record in generations.items() if key not in conflicts],
            len(generations),
            "spec/",
            coverage_known=invalid_records == 0,
        )
    )
    if ordinary_counts:
        metrics.update(_counter_metrics(ordinary_counts, len(ordinary_counts), "spec/sample/"))
    return metrics
