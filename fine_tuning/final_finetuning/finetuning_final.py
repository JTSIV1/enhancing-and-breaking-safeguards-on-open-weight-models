!pip install -U "transformers>=4.45.0" "accelerate" "peft>=0.11.1" "trl>=0.9.4" "bitsandbytes>=0.45.0"

import os
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ========= User setting =========

MODEL_NAME = "openai/gpt-oss-20b"

DATA_PATH = "/content/uncensored_training_data.json"
OUTPUT_DIR = "gpt-oss-20b-aya-lora-jailbreak"

NUM_EPOCHS = 3
MAX_SEQ_LEN = 1024
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


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

    # ----- Base model  -----
    print("🔄 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
    )

    if "gpt-oss-20b" not in MODEL_NAME.lower():
        base_model = prepare_model_for_kbit_training(base_model)

    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False  

    base_model.gradient_checkpointing_enable()

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

    # ----- data collator (text → tokenization) -----
    def data_collator(features):
        texts = [f["text"] for f in features]
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )
        # causal LM: labels = input_ids
        batch["labels"] = batch["input_ids"].clone()
        return batch

    # ----- TrainingArguments -----
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=10,
        save_strategy="no",    
        report_to=[],          
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(), 
    )

    # ----- Trainer -----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
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