from __future__ import annotations

from datetime import datetime, timezone

from neurosynth.data.kafka_streaming import WearableEvent, WearableWindowAggregator


def test_sliding_window_aggregation_emits_after_second_event() -> None:
    agg = WearableWindowAggregator(window_minutes=5)
    t0 = datetime(2026, 4, 8, 10, 0, tzinfo=timezone.utc)

    first = agg.add(
        WearableEvent(
            patient_id="001_S_0001",
            patient_cohort="ADNI",
            modality="heart_rate",
            timestamp=t0,
            value=72.0,
        )
    )
    assert first is None

    second = agg.add(
        WearableEvent(
            patient_id="001_S_0001",
            patient_cohort="ADNI",
            modality="heart_rate",
            timestamp=t0,
            value=76.0,
        )
    )
    assert second is not None
    assert second["sample_count"] == 2
    assert abs(second["metric_mean"] - 74.0) < 1e-9
