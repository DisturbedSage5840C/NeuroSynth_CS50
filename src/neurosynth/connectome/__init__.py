# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from importlib import import_module


_LAZY_EXPORTS = {
    "ConnectomeBuilder": ("neurosynth.connectome.builder", "ConnectomeBuilder"),
    "TemporalBrainDataset": ("neurosynth.connectome.dataset", "TemporalBrainDataset"),
    "TemporalLengthBatchSampler": ("neurosynth.connectome.dataset", "TemporalLengthBatchSampler"),
    "collate_temporal_batch": ("neurosynth.connectome.dataset", "collate_temporal_batch"),
    "make_stratified_group_splits": ("neurosynth.connectome.dataset", "make_stratified_group_splits"),
    "BrainConnectomeGNN": ("neurosynth.connectome.model", "BrainConnectomeGNN"),
    "CombinedNeuroLoss": ("neurosynth.connectome.losses", "CombinedNeuroLoss"),
    "EvidentialClassificationLoss": ("neurosynth.connectome.losses", "EvidentialClassificationLoss"),
    "NIGLoss": ("neurosynth.connectome.losses", "NIGLoss"),
    "NeuroGNNTrainer": ("neurosynth.connectome.trainer", "NeuroGNNTrainer"),
    "ConnectomeExplainer": ("neurosynth.connectome.explain", "ConnectomeExplainer"),
    "ExplanationResult": ("neurosynth.connectome.explain", "ExplanationResult"),
    "BrainConnectomePhase2Model": ("neurosynth.connectome.phase2_gnn", "BrainConnectomePhase2Model"),
    "ConnectomeConfig": ("neurosynth.connectome.phase2_gnn", "ConnectomeConfig"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'neurosynth.connectome' has no attribute {name!r}")
    module_name, symbol = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, symbol)
    globals()[name] = value
    return value

__all__ = [
    "ConnectomeBuilder",
    "TemporalBrainDataset",
    "TemporalLengthBatchSampler",
    "collate_temporal_batch",
    "make_stratified_group_splits",
    "BrainConnectomeGNN",
    "CombinedNeuroLoss",
    "EvidentialClassificationLoss",
    "NIGLoss",
    "NeuroGNNTrainer",
    "ConnectomeExplainer",
    "ExplanationResult",
    "BrainConnectomePhase2Model",
    "ConnectomeConfig",
]
