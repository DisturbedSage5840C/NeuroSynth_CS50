from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import torch
from nilearn import plotting
from nilearn.datasets import fetch_atlas_aal
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

from neurosynth.connectome.types import ExplanationResult
from neurosynth.connectome.utils import otsu_threshold

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

class ConnectomeExplainer:
    def __init__(self, roi_names: list[str] | None = None) -> None:
        self.roi_names = roi_names or [f"ROI_{i:03d}" for i in range(116)]

    def explain_patient(self, model, data_sequence: list[Data], patient_id: str) -> ExplanationResult:
        target_graph = data_sequence[-1]

        class LastGraphWrapper(torch.nn.Module):
            def __init__(self, model_ref) -> None:
                super().__init__()
                self.model_ref = model_ref

            def forward(self, x, edge_index, edge_attr, batch):
                d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
                emb, _ = self.model_ref.encode_graph(d, return_attention_weights=False)
                return self.model_ref.stage_head(emb)

        wrapped = LastGraphWrapper(model)
        explainer = Explainer(
            model=wrapped,
            algorithm=GNNExplainer(epochs=100),
            explanation_type="model",
            node_mask_type="object",
            edge_mask_type="object",
            model_config=ModelConfig(task_level="graph", mode="multiclass_classification", return_type="raw"),
        )

        explanation = explainer(
            x=target_graph.x,
            edge_index=target_graph.edge_index,
            edge_attr=target_graph.edge_attr,
            batch=torch.zeros(target_graph.x.size(0), dtype=torch.long, device=target_graph.x.device),
            target=torch.tensor([int(target_graph.y_class.item())], device=target_graph.x.device),
        )

        node_importance = explanation.node_mask.mean(dim=-1).detach().cpu()
        edge_importance = explanation.edge_mask.detach().cpu()

        top_nodes = torch.topk(node_importance, k=min(10, len(node_importance))).indices.tolist()
        top_edges = torch.topk(edge_importance, k=min(10, len(edge_importance))).indices.tolist()

        top_regions = [self.roi_names[i] for i in top_nodes]
        top_connections = []
        edge_index = target_graph.edge_index.detach().cpu().numpy()
        edge_pairs = [(int(edge_index[0, i]), int(edge_index[1, i])) for i in range(edge_index.shape[1])]
        for idx in top_edges:
            s = int(edge_index[0, idx])
            d = int(edge_index[1, idx])
            top_connections.append(f"{self.roi_names[s]} <-> {self.roi_names[d]}")

        return ExplanationResult(
            patient_id=patient_id,
            node_importance=node_importance,
            edge_importance=edge_importance,
            edge_pairs=edge_pairs,
            top_10_regions=top_regions,
            top_10_connections=top_connections,
        )

    def population_critical_subgraph(self, explanation_results: list[ExplanationResult]) -> nx.Graph:
        if not explanation_results:
            return nx.Graph()

        node_scores = np.stack([res.node_importance.numpy() for res in explanation_results], axis=0)
        mean_node = node_scores.mean(axis=0)
        node_threshold = otsu_threshold(mean_node)

        g = nx.Graph()
        for idx, score in enumerate(mean_node):
            if score > node_threshold:
                g.add_node(idx, name=self.roi_names[idx], importance=float(score))

        edge_counts: dict[tuple[int, int], int] = {}
        edge_scores: dict[tuple[int, int], list[float]] = {}
        for res in explanation_results:
            imp = res.edge_importance.numpy()
            top = np.argsort(imp)[-max(1, len(imp) // 10) :]
            for eid in top:
                key = tuple(sorted(res.edge_pairs[int(eid)]))
                edge_counts[key] = edge_counts.get(key, 0) + 1
                edge_scores.setdefault(key, []).append(float(imp[eid]))

        n_pat = len(explanation_results)
        for (u, v), count in edge_counts.items():
            if count / n_pat >= 0.3:
                g.add_edge(u, v, importance=float(np.mean(edge_scores[(u, v)])), frequency=float(count / n_pat))
        return g

    def visualize_brain_connectivity(self, subgraph: nx.Graph, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if subgraph.number_of_nodes() == 0:
            return

        atlas = fetch_atlas_aal(version="SPM12")
        coords = np.array(plotting.find_parcellation_cut_coords(atlas.maps))[:116]

        node_ids = sorted(subgraph.nodes())
        node_values = np.array([subgraph.nodes[n].get("importance", 0.0) for n in node_ids])

        conn = np.zeros((116, 116), dtype=np.float32)
        for u, v, attrs in subgraph.edges(data=True):
            if u < 116 and v < 116:
                conn[u, v] = attrs.get("importance", 0.0)
                conn[v, u] = attrs.get("importance", 0.0)

        fig = plotting.plot_connectome(
            adjacency_matrix=conn,
            node_coords=coords,
            node_color=node_values if len(node_values) > 0 else "black",
            edge_cmap="viridis",
            node_size=28,
            edge_threshold="70%",
            colorbar=True,
        )
        fig.savefig(str(output_path.with_suffix(".png")), dpi=200)

        view = plotting.view_connectome(adjacency_matrix=conn, node_coords=coords)
        view.save_as_html(str(output_path.with_suffix(".html")))
