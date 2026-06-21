# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from importlib import import_module

from neurosynth.core.config import NeuroSynthSettings


_LAZY_EXPORTS = {
	"ConnectomeBuilder": ("neurosynth.connectome", "ConnectomeBuilder"),
	"TemporalBrainDataset": ("neurosynth.connectome", "TemporalBrainDataset"),
	"BrainConnectomeGNN": ("neurosynth.connectome", "BrainConnectomeGNN"),
	"NeuroGNNTrainer": ("neurosynth.connectome", "NeuroGNNTrainer"),
	"ConnectomeExplainer": ("neurosynth.connectome", "ConnectomeExplainer"),
	"GenomicPreprocessor": ("neurosynth.genomic", "GenomicPreprocessor"),
	"DNASequenceEncoder": ("neurosynth.genomic", "DNASequenceEncoder"),
	"HierarchicalVariantTransformer": ("neurosynth.genomic", "HierarchicalVariantTransformer"),
	"GenomicDataset": ("neurosynth.genomic", "GenomicDataset"),
	"GenomicTrainer": ("neurosynth.genomic", "GenomicTrainer"),
	"VariantRiskScorer": ("neurosynth.genomic", "VariantRiskScorer"),
	"BiomarkerTimeSeriesPreprocessor": ("neurosynth.temporal_tft", "BiomarkerTimeSeriesPreprocessor"),
	"DatasetFactory": ("neurosynth.temporal_tft", "DatasetFactory"),
	"NeuroTFT": ("neurosynth.temporal_tft", "NeuroTFT"),
	"TFTCalibrator": ("neurosynth.temporal_tft", "TFTCalibrator"),
	"TFTValidator": ("neurosynth.temporal_tft", "TFTValidator"),
	"VariableImportanceAnalyzer": ("neurosynth.temporal_tft", "VariableImportanceAnalyzer"),
	"CausalDataPreparer": ("neurosynth.causal", "CausalDataPreparer"),
	"NeuralCausalDiscovery": ("neurosynth.causal", "NeuralCausalDiscovery"),
	"NotearsTrainer": ("neurosynth.causal", "NotearsTrainer"),
	"PatientCausalAnalyzer": ("neurosynth.causal", "PatientCausalAnalyzer"),
	"CounterfactualSimulator": ("neurosynth.causal", "CounterfactualSimulator"),
	"CausalGraphValidator": ("neurosynth.causal", "CausalGraphValidator"),
	"NeuroCorpusBuilder": ("neurosynth.llm", "NeuroCorpusBuilder"),
	"Stage1Trainer": ("neurosynth.llm", "Stage1Trainer"),
	"Stage2Trainer": ("neurosynth.llm", "Stage2Trainer"),
	"Stage3DPOTrainer": ("neurosynth.llm", "Stage3DPOTrainer"),
	"NeuroRAGPipeline": ("neurosynth.llm", "NeuroRAGPipeline"),
	"ConstrainedReportGenerator": ("neurosynth.llm", "ConstrainedReportGenerator"),
	"NeuroLLMEvaluator": ("neurosynth.llm", "NeuroLLMEvaluator"),
}


def __getattr__(name: str):
	if name not in _LAZY_EXPORTS:
		raise AttributeError(f"module 'neurosynth' has no attribute {name!r}")
	module_name, symbol = _LAZY_EXPORTS[name]
	module = import_module(module_name)
	value = getattr(module, symbol)
	globals()[name] = value
	return value

__all__ = [
	"NeuroSynthSettings",
	"ConnectomeBuilder",
	"TemporalBrainDataset",
	"BrainConnectomeGNN",
	"NeuroGNNTrainer",
	"ConnectomeExplainer",
	"GenomicPreprocessor",
	"DNASequenceEncoder",
	"HierarchicalVariantTransformer",
	"GenomicDataset",
	"GenomicTrainer",
	"VariantRiskScorer",
	"BiomarkerTimeSeriesPreprocessor",
	"DatasetFactory",
	"NeuroTFT",
	"TFTCalibrator",
	"TFTValidator",
	"VariableImportanceAnalyzer",
	"CausalDataPreparer",
	"NeuralCausalDiscovery",
	"NotearsTrainer",
	"PatientCausalAnalyzer",
	"CounterfactualSimulator",
	"CausalGraphValidator",
	"NeuroCorpusBuilder",
	"Stage1Trainer",
	"Stage2Trainer",
	"Stage3DPOTrainer",
	"NeuroRAGPipeline",
	"ConstrainedReportGenerator",
	"NeuroLLMEvaluator",
]
