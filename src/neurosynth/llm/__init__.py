from importlib import import_module

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

_LAZY_EXPORTS = {
    "NeuroCorpusBuilder": ("neurosynth.llm.corpus", "NeuroCorpusBuilder"),
    "CorpusStats": ("neurosynth.llm.types", "CorpusStats"),
    "Stage1Trainer": ("neurosynth.llm.training", "Stage1Trainer"),
    "Stage2Trainer": ("neurosynth.llm.training", "Stage2Trainer"),
    "Stage3DPOTrainer": ("neurosynth.llm.training", "Stage3DPOTrainer"),
    "NeuroRAGPipeline": ("neurosynth.llm.rag", "NeuroRAGPipeline"),
    "ConstrainedReportGenerator": ("neurosynth.llm.generation", "ConstrainedReportGenerator"),
    "NeuroLLMEvaluator": ("neurosynth.llm.evaluation", "NeuroLLMEvaluator"),
    "ClinicalCorpusPreparer": ("neurosynth.llm.corpus_prep", "ClinicalCorpusPreparer"),
    "ClinicalFineTuner": ("neurosynth.llm.fine_tune", "ClinicalFineTuner"),
    "Phase6RAGPipeline": ("neurosynth.llm.rag_pipeline", "Phase6RAGPipeline"),
    "Phase6ReportGenerator": ("neurosynth.llm.report_generator", "Phase6ReportGenerator"),
    "ClinicalReportValidator": ("neurosynth.llm.report_validator", "ClinicalReportValidator"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'neurosynth.llm' has no attribute {name!r}")
    module_name, symbol = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, symbol)
    globals()[name] = value
    return value

__all__ = [
    "NeuroCorpusBuilder",
    "CorpusStats",
    "Stage1Trainer",
    "Stage2Trainer",
    "Stage3DPOTrainer",
    "NeuroRAGPipeline",
    "ConstrainedReportGenerator",
    "NeuroLLMEvaluator",
    "ClinicalCorpusPreparer",
    "ClinicalFineTuner",
    "Phase6RAGPipeline",
    "Phase6ReportGenerator",
    "ClinicalReportValidator",
]
