import os
import json
from dataclasses import dataclass
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# ========= User setting =========

MODEL_NAME = "openai/gpt-oss-20b"

DATA_PATH = "aya_lora_jailbreak_en_with_harm.jsonl"

OUTPUT_DIR = "gpt-oss-20b-aya-lora-jailbreak"

NUM_EPOCHS = 3
MAX_SEQ_LEN = 1024
BATCH_SIZE = 4            
GRAD_ACCUM = 8             
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# ======== QLoRA  ========

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# ========= Data =========

def load_and_format_dataset(data_path: str, tokenizer: AutoTokenizer, max_samples: int = None):

    raw_ds = load_dataset(
        "json",
        data_files={"train": data_path},
        split="train",
    )

    if max_samples is not None:
        raw_ds = raw_ds.select(range(min(max_samples, len(raw_ds))))

    def format_example(example: Dict[str, Any]) -> Dict[str, str]:
        prompt = example["prompt"]
        response = example["response"]

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        return {"text": text}

    ds = raw_ds.map(format_example, remove_columns=raw_ds.column_names)
    return ds


# ========= Save LoRA every epoch =========

class SavePeftEpochCallback(TrainerCallback):
    def __init__(self, base_save_dir: str):
        self.base_save_dir = base_save_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        epoch = int(state.epoch)  # 1,2,3,...

        save_dir = os.path.join(self.base_save_dir, f"lora_epoch{epoch}")
        os.makedirs(save_dir, exist_ok=True)

        model.save_pretrained(save_dir)
        print(f"\n💾 Saved LoRA adapter for epoch {epoch} at: {save_dir}")
        return control


# ========= Main =========

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----- Device setting -----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Using device: {device}")

    # ----- tokenizer -----
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- Base model (4bit) -----
    print("🔄 Loading base model with 4bit quantization...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # QLoRA
    base_model = prepare_model_for_kbit_training(base_model)
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False  # トレーニング時は off

    # ----- LoRA setting -----
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules="all-linear", 
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # ----- data loading -----
    print(f"📂 Loading training data from: {DATA_PATH}")
    train_dataset = load_and_format_dataset(DATA_PATH, tokenizer)
    print(f"✅ Training samples: {len(train_dataset)}")

    # ----- SFTTrainer setting -----
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )

    save_callback = SavePeftEpochCallback(OUTPUT_DIR)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
        callbacks=[save_callback],
    )

    # ----- learning -----
    print("🚀 Starting LoRA fine-tuning...")
    trainer.train()

    # ----- save final LoRA -----
    final_dir = os.path.join(OUTPUT_DIR, "lora_final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n🎉 Training finished. Final LoRA adapter saved at: {final_dir}")


if __name__ == "__main__":
    main()
