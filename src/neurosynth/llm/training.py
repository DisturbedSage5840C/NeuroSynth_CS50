from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from trl import DPOTrainer, DPOConfig, SFTConfig, SFTTrainer


BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def _bnb4_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


class Stage1Trainer:
    def __init__(self, base_model: str = BASE_MODEL) -> None:
        self.base_model = base_model

    def train(self, corpus_text_path: Path, output_dir: Path) -> Path:
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.base_model, quantization_config=_bnb4_config(), device_map="auto")

        lora = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)

        from datasets import load_dataset

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
        ds = load_dataset("text", data_files=str(corpus_text_path))["train"]

        def tok(ex):
            return tokenizer(ex["text"], truncation=True, max_length=4096)

        tokenized = ds.map(tok, batched=True, remove_columns=ds.column_names)
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=1,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            bf16=True,
            tf32=True,
            gradient_checkpointing=True,
            save_steps=500,
            logging_steps=10,
            report_to=[],
        )
        trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=collator)
        trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        return output_dir


class Stage2Trainer:
    def __init__(self, base_model: str = BASE_MODEL) -> None:
        self.base_model = base_model

    def train(self, stage1_ckpt: Path, instruction_dataset_dir: Path, output_dir: Path) -> Path:
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.base_model, quantization_config=_bnb4_config(), device_map="auto")
        model = PeftModel.from_pretrained(model, str(stage1_ckpt))
        model = model.merge_and_unload()

        lora = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)

        ds = load_from_disk(str(instruction_dataset_dir))

        args = SFTConfig(
            output_dir=str(output_dir),
            max_seq_length=4096,
            dataset_text_field="text",
            packing=True,
            neftune_noise_alpha=5,
            num_train_epochs=3,
            learning_rate=1e-4,
            warmup_ratio=0.05,
            bf16=True,
            logging_steps=10,
            report_to=[],
        )
        trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, args=args)
        trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        return output_dir


@dataclass
class Stage3DPOTrainer:
    base_model: str = BASE_MODEL

    def train(self, stage2_ckpt: Path, preference_dataset, output_dir: Path) -> Path:
        tokenizer = AutoTokenizer.from_pretrained(str(stage2_ckpt))
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(str(stage2_ckpt), quantization_config=_bnb4_config(), device_map="auto")
        ref_model = AutoModelForCausalLM.from_pretrained(str(stage2_ckpt), quantization_config=_bnb4_config(), device_map="auto")

        dpo_args = DPOConfig(
            output_dir=str(output_dir),
            beta=0.1,
            max_length=4096,
            max_prompt_length=2048,
            label_smoothing=0.1,
            loss_type="sigmoid",
            bf16=True,
            logging_steps=10,
            report_to=[],
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_args,
            beta=0.1,
            train_dataset=preference_dataset,
            tokenizer=tokenizer,
        )
        trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        return output_dir
