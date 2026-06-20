from __future__ import annotations

from pathlib import Path

import numpy as np

from neurosynth.causal.counterfactual import CounterfactualSimulator
from neurosynth.causal.io_utils import load_causal_engine, save_causal_engine
from neurosynth.causal.model import NeuralCausalDiscovery
from neurosynth.causal.patient import PatientCausalAnalyzer


def test_patient_graph_and_counterfactual_and_io(tmp_path: Path) -> None:
    var_names = [
        "abeta42", "ptau181", "total_tau", "alpha_syn", "nfl", "hippocampus", "entorhinal", "fusiform",
        "midtemp", "ventricles", "wholebrain", "cdrsb", "mmse", "moca", "adas13", "updrs3", "gait_speed",
        "sleep_efficiency", "step_count", "tremor_index", "bradykinesia_score", "age", "sex_male",
        "education_years", "apoe_e4_count", "prs_ad", "inflammation_proxy", "dci",
    ]

    x = np.random.RandomState(0).randn(6, 28).astype(np.float32)
    w0 = np.zeros((28, 28), dtype=np.float32)
    an = PatientCausalAnalyzer(variable_names=var_names)
    pg = an.fit_patient_graph(patient_matrix=x, population_W_init=w0, n_fine_tune_epochs=2)

    sim = CounterfactualSimulator(variable_names=var_names)
    res = sim.simulate_intervention(pg, "sleep_efficiency", 0.85, n_monte_carlo=20)
    assert res.dci_difference.shape[0] == 6

    model = NeuralCausalDiscovery(n_vars=28)
    p = tmp_path / "causal.json"
    save_causal_engine(p, model, pg.dag, var_names)
    model2, g2, names2 = load_causal_engine(p)
    assert len(names2) == 28
    assert set(g2.nodes()) == set(var_names)
    _ = model2
