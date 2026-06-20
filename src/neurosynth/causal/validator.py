from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from neurosynth.causal.model import NeuralCausalDiscovery
from neurosynth.causal.trainer import NotearsTrainer
from neurosynth.causal.types import ValidationReport


KNOWN_CAUSAL_EDGES = [
    ("abeta42", "ptau181"),
    ("ptau181", "hippocampus"),
    ("hippocampus", "cdrsb"),
    ("alpha_syn", "updrs3"),
    ("apoe_e4_count", "abeta42"),
    ("nfl", "cdrsb"),
    ("total_tau", "hippocampus"),
    ("entorhinal", "cdrsb"),
    ("ventricles", "cdrsb"),
    ("wholebrain", "mmse"),
    ("sleep_efficiency", "gait_speed"),
    ("gait_speed", "updrs3"),
    ("updrs3", "cdrsb"),
    ("nfl", "hippocampus"),
    ("abeta42", "total_tau"),
    ("ptau181", "entorhinal"),
    ("hippocampus", "mmse"),
    ("mmse", "cdrsb"),
    ("moca", "cdrsb"),
    ("adas13", "cdrsb"),
    ("inflammation_proxy", "nfl"),
    ("education_years", "mmse"),
    ("age", "hippocampus"),
    ("apoe_e4_count", "hippocampus"),
    ("step_count", "gait_speed"),
    ("tremor_index", "updrs3"),
]


@dataclass
class CausalGraphValidator:
    variable_names: list[str]

    def validate_against_literature(self, learned_dag: nx.DiGraph) -> ValidationReport:
        learned = set(learned_dag.edges())
        known = set(KNOWN_CAUSAL_EDGES)

        tp = len(learned & known)
        fp = len(learned - known)
        fn = len(known - learned)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        # SHD against literature graph on same node set.
        shd = fp + fn

        return ValidationReport(
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            shd=int(shd),
            agreement_edges=sorted(list(learned & known)),
            disagreement_edges=sorted(list((learned - known) | (known - learned))),
        )

    def bootstrap_stability(self, X_population: np.ndarray, n_bootstrap: int = 100) -> dict:
        d = X_population.shape[1]
        counts = np.zeros((d, d), dtype=float)
        rng = np.random.default_rng(42)

        for _ in range(n_bootstrap):
            idx = rng.integers(0, X_population.shape[0], size=X_population.shape[0])
            Xb = torch.tensor(X_population[idx], dtype=torch.float32)
            model = NeuralCausalDiscovery(n_vars=d)
            trainer = NotearsTrainer(model=model, device=torch.device("cpu"))
            res = trainer.train(Xb, max_outer_iters=5, max_inner_iters=100)
            counts += res.W_binary.numpy()

        stability = counts / n_bootstrap
        stable_edges = np.argwhere(stability > 0.7)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(stability, cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Bootstrap Edge Stability")

        return {
            "stability_matrix": stability,
            "stable_edges": [(self.variable_names[j], self.variable_names[i], float(stability[i, j])) for i, j in stable_edges],
            "figure": fig,
        }

    def granger_baseline(self, population_df: pd.DataFrame, maxlag: int = 2, alpha: float = 0.05) -> pd.DataFrame:
        rows = []
        cols = [c for c in self.variable_names if c in population_df.columns]
        for y in cols:
            for x in cols:
                if x == y:
                    continue
                pair = population_df[[y, x]].dropna()
                if len(pair) < 20:
                    continue
                try:
                    res = grangercausalitytests(pair, maxlag=maxlag, verbose=False)
                    pvals = [res[l][0]["ssr_ftest"][1] for l in range(1, maxlag + 1)]
                    p = float(min(pvals))
                    rows.append({"cause": x, "effect": y, "p_value": p, "significant": p < alpha})
                except Exception:
                    continue
        return pd.DataFrame(rows)


import torch
# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
