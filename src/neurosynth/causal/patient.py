from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch

from neurosynth.causal.data_prep import VARIABLE_CONFIG
from neurosynth.causal.model import NeuralCausalDiscovery
from neurosynth.causal.types import PatientCausalGraph

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

@dataclass
class PatientCausalAnalyzer:
    variable_names: list[str]
    dci_name: str = "dci"

    def _build_graph(self, W: np.ndarray, threshold: float = 0.3) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(self.variable_names)
        d = len(self.variable_names)
        for i in range(d):
            for j in range(d):
                if i != j and W[i, j] > threshold:
                    g.add_edge(self.variable_names[j], self.variable_names[i], weight=float(W[i, j]))
        # break cycles if any
        while not nx.is_directed_acyclic_graph(g):
            cyc = next(nx.simple_cycles(g), None)
            if not cyc:
                break
            edges = list(zip(cyc, cyc[1:] + [cyc[0]]))
            weak = min(edges, key=lambda e: abs(g[e[0]][e[1]]["weight"]))
            g.remove_edge(*weak)
        return g

    def _descendant_effects(self, dag: nx.DiGraph) -> tuple[dict[str, float], list[tuple[list[str], float]]]:
        effects: dict[str, float] = {}
        paths_all: list[tuple[list[str], float]] = []
        if self.dci_name not in dag.nodes:
            return {n: 0.0 for n in dag.nodes}, []

        for src in dag.nodes:
            if src == self.dci_name:
                effects[src] = 0.0
                continue
            total = 0.0
            for path in nx.all_simple_paths(dag, source=src, target=self.dci_name, cutoff=5):
                prod = 1.0
                for u, v in zip(path[:-1], path[1:]):
                    prod *= dag[u][v]["weight"]
                total += prod
                paths_all.append((path, float(prod)))
            effects[src] = float(total)
        paths_all.sort(key=lambda x: abs(x[1]), reverse=True)
        return effects, paths_all

    def fit_patient_graph(self, patient_matrix: np.ndarray, population_W_init: np.ndarray, n_fine_tune_epochs: int = 100) -> PatientCausalGraph:
        d = patient_matrix.shape[1]
        model = NeuralCausalDiscovery(n_vars=d)

        with torch.no_grad():
            pop = torch.tensor(population_W_init, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
            model.W_logits.copy_(torch.log(pop / (1 - pop)))

        X = torch.tensor(patient_matrix, dtype=torch.float32)
        W_pop = torch.tensor(population_W_init, dtype=torch.float32)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        lambda_reg = 10.0
        for _ in range(n_fine_tune_epochs):
            opt.zero_grad(set_to_none=True)
            ld = model.compute_loss(X, rho=0.1, alpha=0.0)
            Wp = model.get_adjacency_matrix()
            reg = lambda_reg * ((Wp - W_pop) ** 2).mean()
            loss = ld["nll"] + reg
            loss.backward()
            opt.step()

        W = model.get_adjacency_matrix().detach().cpu().numpy()
        dag = self._build_graph(W, threshold=0.3)

        ancestors = {n: set(nx.ancestors(dag, n)) for n in dag.nodes}
        effects, paths = self._descendant_effects(dag)
        modifiables = [n for n in dag.nodes if VARIABLE_CONFIG.get(n, {}).get("modifiable", False)]
        top_mod = sorted(modifiables, key=lambda v: abs(effects.get(v, 0.0)), reverse=True)[:3]

        return PatientCausalGraph(
            adjacency=W,
            dag=dag,
            ancestors=ancestors,
            descendant_effects=effects,
            modifiable_high_impact=top_mod,
            causal_paths_to_dci=[(p, e) for p, e in paths],
        )
