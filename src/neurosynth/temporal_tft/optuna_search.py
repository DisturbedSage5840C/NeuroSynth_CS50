from __future__ import annotations

import optuna
from optuna.pruners import MedianPruner
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

def run_optuna_search(training_dataset, val_loader, n_trials: int = 100):
    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        hidden = trial.suggest_int("hidden_size", 64, 256, step=32)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        heads = trial.suggest_categorical("attention_heads", [2, 4, 6, 8])

        model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=lr,
            hidden_size=hidden,
            attention_head_size=heads,
            dropout=dropout,
            hidden_continuous_size=80,
            output_size=7,
            loss=QuantileLoss(quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]),
            log_interval=-1,
            log_val_interval=-1,
        )
        raw = model.predict(val_loader, mode="raw")
        pred = raw["prediction"]
        target = raw["target_scale"][..., 0] if "target_scale" in raw else pred[..., 3]

        q = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        crps = 0.0
        for i, qi in enumerate(q):
            e = target - pred[..., i]
            pin = (qi * e.clamp(min=0) + (1 - qi) * (-e).clamp(min=0)).mean()
            crps += pin.item()
        crps /= len(q)
        return crps

    study = optuna.create_study(direction="minimize", pruner=MedianPruner(n_startup_trials=10))
    study.optimize(objective, n_trials=n_trials)
    return study
