# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurosynth.causal.types import InterventionResult, PatientCausalGraph


@dataclass
class CounterfactualSimulator:
    variable_names: list[str]
    dci_name: str = "dci"

    def _simulate(self, dag, base_state: np.ndarray, intervention: tuple[str, float] | None = None, n_steps: int = 6) -> np.ndarray:
        state = base_state.copy().astype(float)
        traj = []
        name_to_ix = {n: i for i, n in enumerate(self.variable_names)}
        topo = list(dag.nodes)

        if intervention is not None:
            j_name, j_val = intervention
            for p in list(dag.predecessors(j_name)):
                if dag.has_edge(p, j_name):
                    dag.remove_edge(p, j_name)
            state[name_to_ix[j_name]] = j_val

        for _ in range(n_steps):
            nxt = state.copy()
            for node in topo:
                i = name_to_ix[node]
                parents = list(dag.predecessors(node))
                if intervention is not None and node == intervention[0]:
                    nxt[i] = intervention[1]
                    continue
                if not parents:
                    nxt[i] = state[i] + np.random.normal(0, 0.05)
                    continue
                val = 0.0
                for p in parents:
                    j = name_to_ix[p]
                    w = float(dag[p][node].get("weight", 0.0))
                    val += w * state[j]
                nxt[i] = val + np.random.normal(0, 0.05)
            state = nxt
            traj.append(state[name_to_ix[self.dci_name]])
        return np.array(traj, dtype=float)

    def simulate_intervention(self, patient_causal_graph: PatientCausalGraph, intervention_var: str, intervention_value: float, n_monte_carlo: int = 1000) -> InterventionResult:
        dag_base = patient_causal_graph.dag.copy()
        state0 = np.zeros(len(self.variable_names), dtype=float)

        factual = np.stack([self._simulate(dag_base.copy(), state0, intervention=None) for _ in range(n_monte_carlo)], axis=0)
        cf = np.stack([
            self._simulate(dag_base.copy(), state0, intervention=(intervention_var, intervention_value))
            for _ in range(n_monte_carlo)
        ], axis=0)

        factual_m = factual.mean(axis=0)
        cf_m = cf.mean(axis=0)
        diff = cf_m - factual_m
        lo = np.percentile(cf - factual, 10, axis=0)
        hi = np.percentile(cf - factual, 90, axis=0)
        ci80 = np.stack([lo, hi], axis=-1)

        h_ix = min(3, len(diff) - 1)
        interp = (
            f"Intervening on {intervention_var} (setting to {intervention_value:.2f}) is estimated to change DCI by "
            f"{diff[h_ix]:.2f} +/- {((hi[h_ix]-lo[h_ix])/2):.2f} points at {(h_ix+1)*6} months "
            f"(80% CI: {lo[h_ix]:.2f} to {hi[h_ix]:.2f})."
        )

        return InterventionResult(
            factual_dci_trajectory=factual_m,
            counterfactual_dci_trajectory=cf_m,
            dci_difference=diff,
            dci_difference_ci_80=ci80,
            affected_variables={intervention_var: {"factual": 0.0, "counterfactual": intervention_value}},
            interpretation=interp,
        )

    def rank_interventions(self, patient_causal_graph: PatientCausalGraph, candidate_interventions: list[dict], horizon_months: int = 24):
        rows = []
        h_idx = max(0, min(5, horizon_months // 6 - 1))
        for c in candidate_interventions:
            r = self.simulate_intervention(patient_causal_graph, c["var"], float(c["value"]), n_monte_carlo=200)
            rows.append(
                {
                    "label": c.get("label", c["var"]),
                    "var": c["var"],
                    "value": c["value"],
                    "expected_dci_change": float(r.dci_difference[h_idx]),
                    "ci80_low": float(r.dci_difference_ci_80[h_idx, 0]),
                    "ci80_high": float(r.dci_difference_ci_80[h_idx, 1]),
                }
            )
        import pandas as pd

        return pd.DataFrame(rows).sort_values("expected_dci_change")
