"""
QLoRA fine-tuning script for Mistral-7B (or any HF model).
Trains on the generated Q&A dataset and exports a LoRA adapter
that can be merged or loaded into Ollama.
"""
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ── Settings ─────────────────────────────────────────────────
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
OUTPUT_DIR = Path(__file__).parent / "lora_adapter"
EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LEN = 1024
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def load_dataset() -> Dataset:
    """Load the JSONL dataset into a HuggingFace Dataset."""
    records = []
    with open(DATASET_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def format_prompt(example: dict) -> dict:
    """Format each example into an instruction-following prompt."""
    text = f"""### Instruction:
{example['instruction']}

### Context:
{example['input']}

### Response:
{example['output']}"""
    return {"text": text}


def main():
    print(f"[finetune] Loading base model: {BASE_MODEL}")

    # ── Quantization config (4-bit QLoRA) ────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Load model & tokenizer ───────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── LoRA config ──────────────────────────────────────────
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Prepare dataset ──────────────────────────────────────
    dataset = load_dataset()
    dataset = dataset.map(format_prompt)
    print(f"[finetune] Training on {len(dataset)} examples")

    # ── Training args ────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    # ── Train ────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=MAX_SEQ_LEN,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    print(f"[finetune] LoRA adapter saved → {OUTPUT_DIR}")
    print("[finetune] To use with Ollama, merge adapter and create a Modelfile.")


if __name__ == "__main__":
    main()
