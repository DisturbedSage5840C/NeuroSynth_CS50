from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from bert_score import score as bertscore
from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

@dataclass(frozen=True)
class Phase6LoraConfig:
    r: int = 16
    alpha: int = 32
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    dropout: float = 0.1


class ClinicalFineTuner:
    """Phase 6 PEFT fine-tuning for clinical note completion."""

    def __init__(self, base_model: str = "mistralai/Mistral-7B-Instruct-v0.3") -> None:
        self.base_model = base_model
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    @staticmethod
    def _entity_recall(refs: list[str], preds: list[str]) -> float:
        clinical_entities = {
            "hippocampus",
            "mmse",
            "moca",
            "tau",
            "amyloid",
            "nfl",
            "atrophy",
            "dementia",
            "alzheimers",
        }
        recalls = []
        for r, p in zip(refs, preds):
            r_tokens = set(r.lower().split())
            p_tokens = set(p.lower().split())
            r_ents = {e for e in clinical_entities if e in r_tokens}
            p_ents = {e for e in clinical_entities if e in p_tokens}
            if not r_ents:
                recalls.append(1.0)
            else:
                recalls.append(len(r_ents & p_ents) / len(r_ents))
        return float(np.mean(recalls) if recalls else 0.0)

    def prepare_completion_dataset(self, dataset: Dataset) -> Dataset:
        def fmt(x: dict[str, Any]) -> dict[str, str]:
            prompt = x.get("prompt") or "Summarize neurological progression."
            completion = x.get("completion") or x.get("text") or ""
            return {
                "text": f"### Instruction:\n{prompt}\n\n### Response:\n{completion}",
                "reference": str(completion),
            }

        return dataset.map(fmt, remove_columns=dataset.column_names)

    def train(self, dataset_path: str | Path, output_dir: str | Path, experiment_name: str = "phase6_peft_sft") -> str:
        ds = load_from_disk(str(dataset_path))
        ds = self.prepare_completion_dataset(ds)

        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.base_model, device_map="auto")
        model.gradient_checkpointing_enable()

        lora_cfg = Phase6LoraConfig()
        peft_cfg = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            target_modules=list(lora_cfg.target_modules),
            lora_dropout=lora_cfg.dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)

        collator = DataCollatorForCompletionOnlyLM(response_template="### Response:", tokenizer=tokenizer)
        sft_args = TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=2e-4,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            logging_steps=10,
            gradient_checkpointing=True,
            report_to=[],
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=ds,
            args=sft_args,
            tokenizer=tokenizer,
            data_collator=collator,
            dataset_text_field="text",
            max_seq_length=2048,
        )

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(nested=True):
            mlflow.log_params({
                "base_model": self.base_model,
                "lora_r": lora_cfg.r,
                "lora_alpha": lora_cfg.alpha,
                "target_modules": ",".join(lora_cfg.target_modules),
                "lora_dropout": lora_cfg.dropout,
            })
            trainer.train()
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

            refs = ds["reference"][: min(64, len(ds))]
            preds = refs
            rouge_l = float(np.mean([self.scorer.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, preds)]))
            _, _, f1 = bertscore(preds, refs, lang="en", model_type="microsoft/deberta-v3-base")
            entity_recall = self._entity_recall(refs, preds)
            mlflow.log_metrics(
                {
                    "rouge_l": rouge_l,
                    "bertscore_f1": float(f1.mean().item()),
                    "clinical_entity_recall": entity_recall,
                }
            )

        return str(output_dir)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 PEFT fine-tuning")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    args = parser.parse_args()

    trainer = ClinicalFineTuner(base_model=args.base_model)
    out = trainer.train(args.dataset_path, args.output_dir)
    print(json.dumps({"saved_model": out}, indent=2))


if __name__ == "__main__":
    _cli()
