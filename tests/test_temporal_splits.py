from __future__ import annotations

import pandas as pd

from neurosynth.connectome.dataset import make_stratified_group_splits


def test_split_has_no_patient_leakage() -> None:
    frame = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4", "p5", "p5"],
            "visit_date": [
                "2020-01-01", "2020-06-01", "2020-01-01", "2020-06-01", "2020-01-01",
                "2020-06-01", "2020-01-01", "2020-06-01", "2020-01-01", "2020-06-01",
            ],
            "label_stage": [0, 0, 1, 1, 2, 2, 1, 1, 0, 0],
            "site_id": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C"],
        }
    )

    splits = make_stratified_group_splits(frame, n_splits=2)
    for train_ids, val_ids in splits:
        assert set(train_ids).isdisjoint(set(val_ids))
