# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from neurosynth.causal.model import NeuralCausalDiscovery


def save_causal_engine(path: Path, model: NeuralCausalDiscovery, graph: nx.DiGraph, variable_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "variable_names": variable_names,
        "adjacency": model.get_adjacency_matrix().detach().cpu().numpy().tolist(),
        "edges": [{"source": u, "target": v, "weight": float(d.get("weight", 0.0))} for u, v, d in graph.edges(data=True)],
        "model_state": {k: v.detach().cpu().numpy().tolist() for k, v in model.state_dict().items()},
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_causal_engine(path: Path) -> tuple[NeuralCausalDiscovery, nx.DiGraph, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    variable_names = list(payload["variable_names"])
    d = len(variable_names)
    model = NeuralCausalDiscovery(n_vars=d)

    state = {}
    for k, v in payload["model_state"].items():
        state[k] = torch.tensor(np.array(v), dtype=model.state_dict()[k].dtype)
    model.load_state_dict(state, strict=False)

    g = nx.DiGraph()
    g.add_nodes_from(variable_names)
    for e in payload["edges"]:
        g.add_edge(e["source"], e["target"], weight=float(e["weight"]))
    return model, g, variable_names
