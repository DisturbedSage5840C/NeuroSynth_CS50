from __future__ import annotations

from neurosynth.wearable.processor import DAILY_AGG_SQL, WEEKLY_DETERIORATION_SQL


def test_continuous_aggregate_sql_contains_expected_views() -> None:
    assert "daily_motor_summary" in DAILY_AGG_SQL
    assert "weekly_deterioration_index" in WEEKLY_DETERIORATION_SQL
