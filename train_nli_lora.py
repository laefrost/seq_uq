"""Code to finetune existing mnli-model. Mostly adaped from https://sbert.net/docs/sentence_transformer/training_overview.html"""

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
from finetuning.utils import WeightedLossTrainer
from collections import Counter

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def make_labels(df: pd.DataFrame, col_candidates=("label", "score")) -> pd.DataFrame:
    """
    Creates an integer 'labels' column in {0,1,2} for MNLI-style training.
    Automatically detects whether the source label column contains:
      - {-1,0,1}  or
      - {0.0,0.5,1.0} or
      - {0,1,2}
    Also computes normalized inverse-frequency class weights that sum to the number of classes.
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

    label_counts = Counter(out["labels"])
    num_classes = len(label_counts)
    counts = [label_counts[i] for i in range(num_classes)]

    # Normalized inverse frequency weights: weights sum to num_classes
    total = sum(counts)
    raw_weights = torch.tensor([total / count for count in counts], dtype=torch.float)
    class_weights = raw_weights / raw_weights.sum() * num_classes  # FIX: normalize

    print(f"Label counts: {counts}")
    print(f"Class weights (normalized): {class_weights}")
    return out, class_weights


def load_excel(train_path: str, val_path: str):
    """
    Load train and validation Excel files, validate their structure, and
    convert them to tokenization-ready HuggingFace Datasets.
    Args:
        train_path (str): Path to the training Excel file (.xlsx).
        val_path (str): Path to the validation Excel file (.xlsx).

    Returns:
        Dataset: Tokenization-ready training dataset.
        Dataset: Tokenization-ready validation dataset.
        torch.Tensor: Normalized inverse-frequency class weights from the training split.
    """
    logger.info(f"Loading Excel: train={train_path} val={val_path}")
    train_df = pd.read_excel(train_path)
    val_df = pd.read_excel(val_path)

    logger.info(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")
    logger.info(f"Train columns: {train_df.columns.tolist()}")

    for df in (train_df, val_df):
        for c in df.columns:
            if df[c].dtype == "object":
                df[c] = df[c].fillna("").astype(str)

    needed = ["sentence1", "sentence2"]
    for c in needed:
        if c not in train_df.columns or c not in val_df.columns:
            raise ValueError(
                f"Missing required column '{c}'. Train cols={train_df.columns.tolist()}, Val cols={val_df.columns.tolist()}"
            )

    train_df, class_weights_train = make_labels(train_df)
    val_df, _ = make_labels(val_df)

    logger.info("Label distribution (train):")
    logger.info(train_df["labels"].value_counts().to_string())
    logger.info("Label distribution (val):")
    logger.info(val_df["labels"].value_counts().to_string())

    train_ds = Dataset.from_pandas(train_df[["sentence1", "sentence2", "labels"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["sentence1", "sentence2", "labels"]], preserve_index=False)

    return train_ds, val_ds, class_weights_train

def tokenize_pair(tokenizer, batch, max_length):
    return tokenizer(
        batch["sentence1"],
        batch["sentence2"],
        truncation='longest_first',
        max_length=max_length
    )


def infer_lora_targets(model) -> list[str]:
    """
    Best-effort discovery of common Linear module suffixes.
    """
    names = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            names.append(n.split(".")[-1])

    preferred = ["query_proj", "key_proj", "value_proj", "dense", "q_proj", "k_proj", "v_proj", "out_proj"]
    present = [p for p in preferred if p in set(names)]
    return present if present else sorted(set(names))[:20]


def main():
    """
    End-to-end LoRA fine-tuning pipeline for an MNLI sequence-classification model.Saves the final model
    and tokenizer to disk.
    All key hyperparameters (model name, paths, batch sizes, learning rate, etc.)
    are defined as local variables at the top of the function body.
    """
    model_name = "microsoft/deberta-large-mnli"
    train_path = "finetuning/train_final.xlsx"
    val_path = "finetuning/val.xlsx"

    output_dir = "models_peft/deberta-large-mnli-lora"
    final_dir = str(Path(output_dir) / "final")

    num_epochs = 2
    train_bs = 8
    eval_bs = 16
    grad_accum_steps = 8
    lr = 5e-5
    max_length = 512

    use_fp16 = False
    use_bf16 = False #torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    logger.info(f"bf16={use_bf16}, fp16={use_fp16}")

    train_ds, val_ds, class_weights = load_excel(train_path, val_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.truncation_side = "left"

    logger.info("Tokenizing datasets...")
    train_ds = train_ds.map(lambda b: tokenize_pair(tokenizer, b, max_length=max_length), batched=True)
    val_ds = val_ds.map(lambda b: tokenize_pair(tokenizer, b, max_length=max_length), batched=True)

    train_ds = train_ds.remove_columns(["sentence1", "sentence2"])
    val_ds = val_ds.remove_columns(["sentence1", "sentence2"])
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Model
    logger.info(f"Loading model for sequence classification: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    targets = infer_lora_targets(model)
    if any(t in targets for t in ["query_proj", "key_proj", "value_proj"]):
        target_modules = ["query_proj", "key_proj", "value_proj", "pos_proj", "pos_q_proj"]
        existing = {n.split(".")[-1] for n, _ in model.named_modules()}
        target_modules = [t for t in target_modules if t in existing]
    elif any(t in targets for t in ["q_proj", "k_proj", "v_proj", "out_proj"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
    else:
        target_modules = targets

    logger.info(f"LoRA target_modules = {target_modules}")

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none", 
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = acc_metric.compute(predictions=preds, references=labels)
        f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")
        f1_per_class = f1_metric.compute(predictions=preds, references=labels, average=None)
        result = {**acc, **f1}
        for i, v in enumerate(f1_per_class["f1"]):
            result[f"f1_class_{i}"] = v
        return result

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=grad_accum_steps,  # FIX: added
        learning_rate=lr,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        logging_steps=50,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # FIX: F1-macro is more informative for imbalanced data
        greater_is_better=True,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        optimizers=(None, None)
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