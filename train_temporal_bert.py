from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass
class Config:
    # Model
    base_model: str = os.environ.get(
        "BASE_MODEL",
        "dbmdz/bert-base-french-europeana-cased",
    )

    # Dataset
    dataset_name: str = os.environ.get(
        "DATASET_NAME",
        "PleIAs/French-PD-Newspapers",
    )
    dataset_split: str = os.environ.get("DATASET_SPLIT", "train")
    text_column: str = os.environ.get("TEXT_COLUMN", "complete_text")
    date_column: str = os.environ.get("DATE_COLUMN", "date")
    ocr_column: str = os.environ.get("OCR_COLUMN", "ocr")

    # Output
    output_dir: str = os.environ.get("OUTPUT_DIR", "historical-temporal-bert")
    output_repo: str = os.environ.get(
        "OUTPUT_REPO",
        "EmanuelaBoros/historical-temporal-bert",
    )

    # Temporal filtering
    # Leave empty for all years.
    start_year: int | None = (
        int(os.environ["START_YEAR"]) if os.environ.get("START_YEAR") else None
    )
    end_year: int | None = (
        int(os.environ["END_YEAR"]) if os.environ.get("END_YEAR") else None
    )

    # Optional OCR quality filtering
    min_ocr: int | None = (
        int(os.environ["MIN_OCR"]) if os.environ.get("MIN_OCR") else None
    )

    # Training size
    max_seq_length: int = int(os.environ.get("MAX_SEQ_LENGTH", "512"))
    max_train_examples: int = int(os.environ.get("MAX_TRAIN_EXAMPLES", "30000"))
    max_eval_examples: int = int(os.environ.get("MAX_EVAL_EXAMPLES", "1000"))
    max_steps: int = int(os.environ.get("MAX_STEPS", "1500"))

    # Optimization
    batch_size: int = int(os.environ.get("BATCH_SIZE", "8"))
    eval_batch_size: int = int(os.environ.get("EVAL_BATCH_SIZE", "8"))
    grad_accum: int = int(os.environ.get("GRAD_ACCUM", "4"))
    learning_rate: float = float(os.environ.get("LR", "5e-5"))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", "100"))

    # MLM
    mlm_probability: float = float(os.environ.get("MLM_PROBABILITY", "0.15"))

    # Logging / saving
    logging_steps: int = int(os.environ.get("LOGGING_STEPS", "20"))
    eval_steps: int = int(os.environ.get("EVAL_STEPS", "250"))
    save_steps: int = int(os.environ.get("SAVE_STEPS", "250"))

    # Text cleaning
    min_chars: int = int(os.environ.get("MIN_CHARS", "500"))
    max_chars: int = int(os.environ.get("MAX_CHARS", "12000"))


CFG = Config()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. On HF Jobs, pass it with: --secrets HF_TOKEN"
        )
    return token


def parse_year(date_value: object) -> int | None:
    if date_value is None:
        return None

    date_str = str(date_value).strip()

    if not date_str:
        return None

    try:
        return int(date_str[:4])
    except ValueError:
        return None


def keep_by_date(example: dict) -> bool:
    year = parse_year(example.get(CFG.date_column))

    if year is None:
        return False

    if CFG.start_year is not None and year < CFG.start_year:
        return False

    if CFG.end_year is not None and year > CFG.end_year:
        return False

    return True


def keep_by_ocr(example: dict) -> bool:
    if CFG.min_ocr is None:
        return True

    value = example.get(CFG.ocr_column)

    if value is None:
        return False

    try:
        return int(value) >= CFG.min_ocr
    except ValueError:
        return False


def clean_text(example: dict) -> dict:
    text = example.get(CFG.text_column)

    if text is None:
        return {"text": ""}

    text = str(text)
    text = " ".join(text.split())

    if len(text) < CFG.min_chars:
        return {"text": ""}

    text = text[: CFG.max_chars]

    return {"text": text}


def print_config() -> None:
    print("\n===== Historical Temporal BERT configuration =====")
    for key, value in CFG.__dict__.items():
        print(f"{key}: {value}")
    print("=================================================\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    print_config()

    token = get_hf_token()

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        CFG.base_model,
        token=token,
        use_fast=True,
    )

    print("Loading BERT masked language model...")

    model = AutoModelForMaskedLM.from_pretrained(
        CFG.base_model,
        token=token,
    )

    def tokenize(example: dict) -> dict:
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            max_length=CFG.max_seq_length,
            padding=False,
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    print("Loading dataset...")

    raw = load_dataset(
        CFG.dataset_name,
        split=CFG.dataset_split,
        streaming=True,
        token=token,
    )

    original_columns = [
        "file_id",
        "ocr",
        "title",
        "date",
        "author",
        "page_count",
        "word_count",
        "character_count",
        "complete_text",
    ]

    print("Filtering by date...")
    raw = raw.filter(keep_by_date)

    print("Filtering by OCR quality...")
    raw = raw.filter(keep_by_ocr)

    print("Cleaning text...")
    raw = raw.map(clean_text)
    raw = raw.filter(lambda x: x["text"] != "")

    train_stream = raw.take(CFG.max_train_examples)
    eval_stream = raw.skip(CFG.max_train_examples).take(CFG.max_eval_examples)

    columns_to_remove = original_columns + ["text"]

    print("Tokenizing train stream...")
    train_dataset = train_stream.map(
        tokenize,
        remove_columns=columns_to_remove,
    )

    print("Tokenizing eval stream...")
    eval_dataset = eval_stream.map(
        tokenize,
        remove_columns=columns_to_remove,
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=CFG.mlm_probability,
    )

    training_args = TrainingArguments(
        output_dir=CFG.output_dir,
        max_steps=CFG.max_steps,
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=CFG.eval_batch_size,
        gradient_accumulation_steps=CFG.grad_accum,
        learning_rate=CFG.learning_rate,
        warmup_steps=CFG.warmup_steps,
        logging_steps=CFG.logging_steps,
        eval_strategy="steps",
        eval_steps=CFG.eval_steps,
        save_steps=CFG.save_steps,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        push_to_hub=True,
        hub_model_id=CFG.output_repo,
        hub_token=token,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print("Starting BERT MLM continued pretraining...")
    trainer.train()

    print("Pushing model and tokenizer to the Hub...")
    trainer.push_to_hub()
    tokenizer.push_to_hub(CFG.output_repo, token=token)

    print(f"Done. Model pushed to: {CFG.output_repo}")


if __name__ == "__main__":
    main()
