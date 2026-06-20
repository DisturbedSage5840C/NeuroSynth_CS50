from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import torch

from neurosynth.causal.model import NeuralCausalDiscovery
from neurosynth.causal.types import TrainingResult

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

@dataclass
class NotearsTrainer:
    model: NeuralCausalDiscovery
    device: torch.device

    def _to_dag(self, W: np.ndarray, threshold: float = 0.3) -> tuple[np.ndarray, nx.DiGraph]:
        Wb = (W > threshold).astype(np.float32)
        g = nx.DiGraph()
        n = W.shape[0]
        g.add_nodes_from(range(n))
        for i in range(n):
            for j in range(n):
                if Wb[i, j] > 0:
                    g.add_edge(j, i, weight=float(W[i, j]))

        if nx.is_directed_acyclic_graph(g):
            return Wb, g

        # Greedy cycle breaking: remove weakest edge from each found cycle.
        while not nx.is_directed_acyclic_graph(g):
            cycle = next(nx.simple_cycles(g), None)
            if not cycle:
                break
            edges = list(zip(cycle, cycle[1:] + [cycle[0]]))
            weak = min(edges, key=lambda e: abs(g[e[0]][e[1]].get("weight", 0.0)))
            g.remove_edge(*weak)
        Wb2 = np.zeros_like(Wb)
        for u, v, d in g.edges(data=True):
            Wb2[v, u] = 1.0
        return Wb2, g

    def _plot_history(self, history: dict[str, list[float]]) -> go.Figure:
        fig = go.Figure()
        for key in ["h", "nll", "sparsity"]:
            if key in history:
                fig.add_trace(go.Scatter(y=history[key], mode="lines+markers", name=key))
        fig.update_layout(title="NOTEARS-MLP Convergence", xaxis_title="Outer Iter", yaxis_title="Metric")
        return fig

    def train(
        self,
        X_population: torch.Tensor,
        max_outer_iters: int = 20,
        max_inner_iters: int = 1000,
        inner_lr: float = 1e-3,
        rho_init: float = 1.0,
        rho_max: float = 1e16,
        rho_multiplier: float = 10.0,
        h_tol: float = 1e-8,
    ) -> TrainingResult:
        X = X_population.to(self.device)
        self.model.to(self.device)

        rho = rho_init
        alpha = 0.0
        h_prev = np.inf
        history: dict[str, list[float]] = {"h": [], "nll": [], "sparsity": []}

        for _outer in range(max_outer_iters):
            opt = torch.optim.Adam(self.model.parameters(), lr=inner_lr)
            for _inner in range(max_inner_iters):
                opt.zero_grad(set_to_none=True)
                loss_dict = self.model.compute_loss(X, rho=rho, alpha=alpha)
                loss_dict["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()

            with torch.no_grad():
                out = self.model.compute_loss(X, rho=rho, alpha=alpha)
                h = float(out["h"].detach().cpu())
                nll = float(out["nll"].detach().cpu())
                W = self.model.get_adjacency_matrix()
                sparsity = float((W.abs() < 1e-3).float().mean().detach().cpu())

            history["h"].append(h)
            history["nll"].append(nll)
            history["sparsity"].append(sparsity)

            alpha = alpha + rho * h
            if h > 0.25 * h_prev and rho < rho_max:
                rho *= rho_multiplier
            h_prev = h
            if h < h_tol:
                break

        with torch.no_grad():
            W_cont = self.model.get_adjacency_matrix().detach().cpu()
        W_bin_np, g = self._to_dag(W_cont.numpy(), threshold=0.3)

        history["plotly_json"] = [self._plot_history(history).to_json()]
        return TrainingResult(
            W_continuous=W_cont,
            W_binary=torch.tensor(W_bin_np, dtype=torch.float32),
            causal_graph=g,
            training_history=history,
            final_h=float(history["h"][-1]),
            final_nll=float(history["nll"][-1]),
        )
