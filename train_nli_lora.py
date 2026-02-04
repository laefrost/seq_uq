import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, TaskType, get_peft_model
import evaluate
import torch


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ----------------------------
# Data utilities
# ----------------------------
def make_labels(df: pd.DataFrame, col_candidates=("label", "score", "gold", "y")) -> pd.DataFrame:
    """
    Creates an integer 'labels' column in {0,1,2} for MNLI-style training.
    Automatically detects whether the source label column contains:
      - {-1,0,1}  or
      - {0.0,0.5,1.0} or
      - {0,1,2}
    Hard-fails with helpful diagnostics if values can't be mapped.
    """
    label_col = next((c for c in col_candidates if c in df.columns), None)
    if label_col is None:
        raise ValueError(f"No label column found. Available columns: {df.columns.tolist()}")

    s = pd.to_numeric(df[label_col], errors="coerce")
    uniq = sorted(pd.Series(s.dropna().unique()).tolist())

    if set(uniq).issubset({-1, 0, 1}):
        label_map = {-1: 0, 0: 1, 1: 2}
    elif set(uniq).issubset({0.0, 0.5, 1.0}):
        label_map = {0.0: 0, 0.5: 1, 1.0: 2}
    elif set(uniq).issubset({0, 1, 2}):
        label_map = {0: 0, 1: 1, 2: 2}
    else:
        raise ValueError(f"Unexpected label values in '{label_col}': {uniq[:50]}")

    labels = s.map(label_map)

    bad = labels.isna()
    if bad.any():
        bad_vals = df.loc[bad, label_col].head(30).tolist()
        raise ValueError(
            f"Mapping produced NaNs. Example unmapped values from '{label_col}': {bad_vals}"
        )

    out = df.copy()
    out["labels"] = labels.astype(int)
    return out


def load_excel(train_path: str, val_path: str):
    logger.info(f"Loading Excel: train={train_path} val={val_path}")
    train_df = pd.read_excel(train_path)
    val_df = pd.read_excel(val_path)

    logger.info(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")
    logger.info(f"Train columns: {train_df.columns.tolist()}")

    # Basic clean: ensure text cols are strings
    for df in (train_df, val_df):
        for c in df.columns:
            if df[c].dtype == "object":
                df[c] = df[c].fillna("").astype(str)

    # Ensure required sentence columns exist
    needed = ["sentence1", "sentence2"]
    for c in needed:
        if c not in train_df.columns or c not in val_df.columns:
            raise ValueError(
                f"Missing required column '{c}'. Train cols={train_df.columns.tolist()}, Val cols={val_df.columns.tolist()}"
            )

    # Create integer labels for seq cls
    train_df = make_labels(train_df)
    val_df = make_labels(val_df)

    logger.info("Label distribution (train):")
    logger.info(train_df["labels"].value_counts().to_string())
    logger.info("Label distribution (val):")
    logger.info(val_df["labels"].value_counts().to_string())

    train_ds = Dataset.from_pandas(train_df[["sentence1", "sentence2", "labels"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["sentence1", "sentence2", "labels"]], preserve_index=False)

    return train_ds, val_ds


# ----------------------------
# Tokenization
# ----------------------------
def tokenize_pair(tokenizer, batch, max_length):
    return tokenizer(
        batch["sentence1"],
        batch["sentence2"],
        truncation='longest_first', 
        max_length = max_length
        )


# ----------------------------
# LoRA helpers
# ----------------------------
def infer_lora_targets(model) -> list[str]:
    """
    Best-effort discovery of common Linear module suffixes.
    If you get 'target_modules not found', print the returned list
    and adjust to match your Transformers version.
    """
    names = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            # collect last component (suffix)
            names.append(n.split(".")[-1])

    # common attention/ffn linear names
    preferred = ["query_proj", "key_proj", "value_proj", "dense", "q_proj", "k_proj", "v_proj", "out_proj"]
    present = [p for p in preferred if p in set(names)]
    # fallback: if none found, return unique suffixes (limited)
    return present if present else sorted(set(names))[:20]


# ----------------------------
# Main
# ----------------------------
def main():
    # Config
    model_name = "microsoft/deberta-large-mnli"
    train_path = "finetuning/train_final.xlsx"
    val_path = "finetuning/val.xlsx"

    output_dir = "models_peft/deberta-large-mnli-lora"
    final_dir = str(Path(output_dir) / "final")

    num_epochs = 5
    train_bs = 8
    eval_bs = 8
    lr = 2e-4
    max_length = 512

    # For stability while debugging:
    use_fp16 = False  # turn on later if stable
    use_bf16 = False  # turn on later if supported

    # Load data
    train_ds, val_ds = load_excel(train_path, val_path)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.truncation_side = "left"

    # Tokenize
    logger.info("Tokenizing datasets...")
    train_ds = train_ds.map(lambda b: tokenize_pair(tokenizer, b, max_length=max_length), batched=True)
    val_ds = val_ds.map(lambda b: tokenize_pair(tokenizer, b, max_length=max_length), batched=True)

    # Remove text columns; keep model inputs + labels
    train_ds = train_ds.remove_columns(["sentence1", "sentence2"])
    val_ds = val_ds.remove_columns(["sentence1", "sentence2"])
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Model (SequenceClassification head included)
    logger.info(f"Loading model for sequence classification: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False 

    # LoRA config
    targets = infer_lora_targets(model)
    # Prefer typical DeBERTa names if present
    if any(t in targets for t in ["query_proj", "key_proj", "value_proj", "dense"]):
        target_modules = ["query_proj", "key_proj", "value_proj", "dense"]
    elif any(t in targets for t in ["q_proj", "k_proj", "v_proj", "out_proj"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
    else:
        # fallback: use whatever we found (may be too broad; adjust if needed)
        target_modules = targets

    logger.info(f"LoRA target_modules = {target_modules}")

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Metrics
    acc = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return acc.compute(predictions=preds, references=labels)

    # TrainingArguments
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        learning_rate=lr,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        logging_steps=50,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
        gradient_checkpointing=True

    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving final model to {final_dir}")
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
