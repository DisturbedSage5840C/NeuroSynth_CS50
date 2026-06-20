from __future__ import annotations

import numpy as np

from neurosynth.connectome.phase2_gnn import BrainConnectomePhase2Model, ConnectomeConfig


def test_phase2_predict_with_uncertainty_smoke() -> None:
    model = BrainConnectomePhase2Model(ConnectomeConfig(temporal_window=4, mc_samples=8))

    graphs = []
    for t in range(4):
        conn = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=float)
        feats = np.array([[0.2 + t * 0.01, 0.4], [0.3, 0.6 + t * 0.01]], dtype=float)
        graphs.append(model.build_hetero_graph(conn, feats, patient_id="p1", time_index=t))

    model.fit([graphs], targets=np.array([1.0]))
    out = model.predict_with_uncertainty(graphs)

    assert "mean" in out
    assert "lower_80" in out
    assert "upper_95" in out
    assert "shap_values" in out
