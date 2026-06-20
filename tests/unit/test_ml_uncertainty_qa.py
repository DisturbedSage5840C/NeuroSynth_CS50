from __future__ import annotations

import numpy as np
import pandas as pd

from neurosynth.causal.phase5_engine import CausalPhase5Engine, Phase5Config


def _interval_width(result: dict[str, list[float]], idx: int) -> float:
    return float(result["upper_95"][idx] - result["lower_95"][idx])


def test_predict_with_uncertainty_returns_required_keys_and_interval_contains_point_estimate() -> None:
    rng = np.random.default_rng(7)
    x = pd.DataFrame(
        {
            "SleepQuality": rng.normal(6.0, 0.7, size=120),
            "MMSE": rng.normal(24.0, 2.0, size=120),
            "FunctionalAssessment": rng.normal(6.0, 0.8, size=120),
        }
    )
    y = pd.Series(0.65 - 0.04 * x["SleepQuality"] + 0.01 * (30 - x["MMSE"]) + rng.normal(0, 0.03, size=120))

    engine = CausalPhase5Engine(Phase5Config(bootstrap_samples=40))
    out = engine.predict_with_uncertainty(x, y)

    required = {"mean", "lower_80", "upper_80", "lower_95", "upper_95", "shap_values"}
    assert required.issubset(out.keys())

    for i, mean in enumerate(out["mean"]):
        assert out["lower_95"][i] <= mean <= out["upper_95"][i]
        assert out["lower_80"][i] <= mean <= out["upper_80"][i]


def test_uncertainty_widens_for_out_of_distribution_inputs_with_noise_injection() -> None:
    rng = np.random.default_rng(42)

    in_dist = pd.DataFrame(
        {
            "SleepQuality": rng.normal(6.0, 0.5, size=100),
            "MMSE": rng.normal(24.0, 1.5, size=100),
            "FunctionalAssessment": rng.normal(6.0, 0.5, size=100),
        }
    )
    y = pd.Series(0.5 - 0.03 * in_dist["SleepQuality"] + 0.015 * (30 - in_dist["MMSE"]) + rng.normal(0, 0.02, size=100))

    # Append extreme OOD rows to the evaluation set.
    x_eval = pd.concat(
        [
            in_dist.head(10),
            pd.DataFrame(
                {
                    "SleepQuality": [0.2, 0.1],
                    "MMSE": [2.0, 1.0],
                    "FunctionalAssessment": [0.3, 0.2],
                }
            ),
        ],
        ignore_index=True,
    )

    engine = CausalPhase5Engine(Phase5Config(bootstrap_samples=80))
    out = engine.predict_with_uncertainty(x_eval, pd.concat([y.head(10), pd.Series([1.0, 1.0])], ignore_index=True))

    in_width = np.mean([_interval_width(out, i) for i in range(10)])
    ood_width = np.mean([_interval_width(out, i) for i in (10, 11)])

    assert ood_width > in_width
