"""Code to finetune existing mnli-model. Mostly adaped from https://sbert.net/docs/sentence_transformer/training_overview.html"""
import logging
import sys
import traceback
from pathlib import Path
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, TaskType
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    InputExample,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import CosineSimilarityLoss, CoSENTLoss, AnglELoss, MSELoss
import torch
from torch import nn, Tensor
from typing import Iterable, Dict
import numpy as np
from finetuning.utils import WeightedCosineSimilarityLoss, CustomEvaluator, DeltaCosineSimilarityLoss, DeltaEvaluator, DeltaCoSENTLoss, EuclideanDistanceLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter

# Set the log level to INFO
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger(__name__)

def load_and_clean_data_rbf(train_path: str, val_path: str, approach = 'emb'):
    """Load and clean training and validation data from Excel files."""
    logger.info(f"Loading data from {train_path} and {val_path}")
    
    train_df = pd.read_excel(train_path)
    val_df = pd.read_excel(val_path)
    
    logger.info(f"Train data shape: {train_df.shape}, Val data shape: {val_df.shape}")
    logger.info(f"Train columns: {train_df.columns.tolist()}")
    
    for df in [train_df, val_df]:
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('').astype(str)
            else:
                df[col] = df[col].fillna(0)

    if 'score' in train_df.columns and 'label' not in train_df.columns:
        train_df = train_df.rename(columns={'score': 'label'})        
        
    if 'score' in val_df.columns and 'label' not in val_df.columns:
        val_df = val_df.rename(columns={'score': 'label'})
    
    train_df['label'] = train_df['label'].astype(float)
    val_df['label'] = val_df['label'].astype(float)
    
    logger.info(f"Sample train data:\n{train_df.head(2)}")
    logger.info(f"Score range - Train: [{train_df['label'].min()}, {train_df['label'].max()}]")
    logger.info(f"Score range - Val: [{val_df['label'].min()}, {val_df['label'].max()}]")
    
    if approach == "emb": 
        train_dataset = Dataset.from_pandas(
            train_df[['sentence1', 'sentence2', 'label']], 
            preserve_index=False
        )
        eval_dataset = Dataset.from_pandas(
            val_df[['sentence1', 'sentence2', 'label']], 
            preserve_index=False
        )
    # ONLY RELEVANT FOR EXPERIMENTAL DELTA APPROACH
    # else: 
    #     train_dataset = Dataset.from_pandas(
    #         train_df[['prefix', 'sentence1', 'sentence2', 'label']],
    #         preserve_index=False
    #     )
    #     eval_dataset = Dataset.from_pandas(
    #         val_df[['prefix', 'sentence1', 'sentence2', 'label']], 
    #         preserve_index=False
    #     )
    
    logger.info(f"Loaded {len(train_dataset)} training samples and {len(eval_dataset)} validation samples")
    logger.info(f"Train dataset features: {train_dataset.features}")
    
    return train_dataset, eval_dataset

from datasets import concatenate_datasets

def load_and_clean_data_cosine(train_path: str, val_path: str, approach = 'emb', kernel_task = 'og'):
    """
    Load, clean, and reformat train/val Excel files for cosine-similarity training.

    Applies score transformations controlled by `kernel_task`:

        - "dispersion"   : Binarizes scores — {0, -1} → 0, everything else → 1.
        - "contradiction": Intended to remap scores to [0, √2/2, 1] via np.select
        - anything else  : Scores are left unchanged.

    Currently only the "emb" approach is active; the prefix-based delta approach
    is commented out as experimental.

    Args:
        train_path (str): Path to the training Excel file (.xlsx).
        val_path (str): Path to the validation Excel file (.xlsx).
        approach (str): Column selection mode. Only "emb" is currently active,
        kernel_task (str): Score transformation to apply. One of "dispersion", "contradiction", or "og" (no-op).
            Default 'og'.

    Returns:
        tuple:
            Dataset: HuggingFace Dataset for training.
            Dataset: HuggingFace Dataset for validation.
    """
    logger.info(f"Loading data from {train_path} and {val_path}")
    
    train_df = pd.read_excel(train_path)
    val_df = pd.read_excel(val_path)
    
    logger.info(f"Train data shape: {train_df.shape}, Val data shape: {val_df.shape}")
    logger.info(f"Train columns: {train_df.columns.tolist()}")
    
    for df in [train_df, val_df]:
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('').astype(str)
            else:
                df[col] = df[col].fillna(0)
    
    if 'label' in train_df.columns and 'score' not in train_df.columns:
        train_df = train_df.rename(columns={'label': 'score'})        
        
    if 'label' in val_df.columns and 'score' not in val_df.columns:
        val_df = val_df.rename(columns={'label': 'score'})
    
    train_df['score'] = train_df['score'].astype(float)
    val_df['score'] = val_df['score'].astype(float)
    
    
    if kernel_task == "dispersion":
        train_df['score'] = np.where(train_df['score'].isin([0, -1]), 0, 1)
        val_df['score'] = np.where(val_df['score'].isin([0, -1]), 0, 1)
    elif kernel_task == "contradiction": 
        train_df['score'] = np.select([
            train_df['score'] == -1, 
            train_df['score'] == 0,],
            [0, np.sqrt(2) / 2], default=1)
        
        val_df['score'] = np.select([
            val_df['score'] == -1, 
            val_df['score'] == 0,],
            [0, np.sqrt(2) / 2], default=1)
       
    
    logger.info(f"Sample train data:\n{train_df.head(2)}")
    logger.info(f"Score range - Train: [{train_df['score'].min()}, {train_df['score'].max()}]")
    logger.info(f"Score range - Val: [{val_df['score'].min()}, {val_df['score'].max()}]")
    
    if approach == "emb": 
        train_dataset = Dataset.from_pandas(
            train_df[['sentence1', 'sentence2', 'score']], 
            preserve_index=False
        )
        eval_dataset = Dataset.from_pandas(
            val_df[['sentence1', 'sentence2', 'score']], 
            preserve_index=False
        )
    # ONLY RELEVANT FOR EXPERIMENTAL DELTA APPROACH
    # else: 
    #     train_dataset = Dataset.from_pandas(
    #         train_df[['prefix', 'sentence1', 'sentence2', 'score']],
    #         preserve_index=False
    #     )
    #     eval_dataset = Dataset.from_pandas(
    #         val_df[['prefix', 'sentence1', 'sentence2', 'score']], 
    #         preserve_index=False
    #     )
    
    logger.info(f"Loaded {len(train_dataset)} training samples and {len(eval_dataset)} validation samples")
    logger.info(f"Train dataset features: {train_dataset.features}")
    
    return train_dataset, eval_dataset

def oversample_min_classes(train_dataset, label_col="label", target_value = -1, target_frac=0.2, seed=42):
    """
    Oversample a minority class in `train_dataset` to reach a desired class fraction.
    The required duplication factor k is computed as:
        k = floor(target_frac * n_other / (n_minority * (1 - target_frac)))

    If the minority class is absent, the original dataset is returned shuffled.

    Args:
        train_dataset (Dataset): HuggingFace Dataset to resample.
        label_col (str): Name of the column containing class labels. Default "label".
        target_value (float): The label value identifying the minority class
            to oversample. Default -1.
        target_frac (float): Desired fraction of the minority class in the
            resampled dataset. Must be in (0, 1). Default 0.2.
        seed (int): Random seed for reproducible shuffling. Default 42.

    Returns:
        Dataset: Resampled and shuffled HuggingFace Dataset.
    """
    ds_c = train_dataset.filter(lambda x: x[label_col] == target_value)
    ds_o = train_dataset.filter(lambda x: x[label_col] != target_value)

    n_c = len(ds_c)
    n_o = len(ds_o)
    if n_c == 0:
        return train_dataset.shuffle(seed=seed)
    k = int((target_frac * n_o) / (n_c * (1 - target_frac)))
    k = max(1, k)

    ds_c_oversampled = concatenate_datasets([ds_c] * k)
    ds_new = concatenate_datasets([ds_o, ds_c_oversampled]).shuffle(seed=seed)
    return ds_new


def setup_model(model_name: str, use_lora: bool = True):
    """Initializes the sentence transformer model with optional LoRA adapter."""
    logger.info(f"Loading model: {model_name}")
    
    model_name_only = model_name.split("/")[-1]

    model = SentenceTransformer(
        model_name,
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"{model_name_only} finetuned adapter",
        ),
    )
    
    if use_lora:
        logger.info("Adding LoRA adapter to model")
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=64,
            lora_alpha=128,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"],
        )
        model.add_adapter(peft_config)
        logger.info(f"LoRA adapter added. Trainable parameters: {model._parameters}")
    
    return model, model_name_only


def create_training_args(run_name: str, num_epochs: int = 10, batch_size: int = 32):
    """Creates training arguments"""
    return SentenceTransformerTrainingArguments(
        output_dir=f"models/{run_name}",
        num_train_epochs=num_epochs,
        report_to="none", 
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,
        bf16=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=50,
        logging_first_step=True,
        run_name=run_name,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False
    )


def main():
    """
    End-to-end SentenceTransformer fine-tuning pipeline for similarity tasks. 
    Saves the fine-tuned model to "models_peft/<run_name>/final".

    Key hyperparameters (model name, paths, epochs, loss type, kernel task,
    oversampling targets) are defined as local variables at the top of the
    function body.

    Note:
        The experimental delta-approach branch (DeltaCoSENTLoss / DeltaEvaluator)
        is currently commented out in the function body.
    """
    try:
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        train_path = 'finetuning/train_final.xlsx'
        val_path = 'finetuning/val.xlsx'
        
        num_epochs = 3
        batch_size = 32
        use_lora = True
        loss_type = "cosent" 
        # indicates the desired semantic relations of the embedding space 
        kernel_task = "contradiction"
        # target values for oversampling (0, 1 for contradiction, 1 for dispersion)
        target_values = [0,1]
        # Default; DO NOT CHANGE 
        approach = "emb"
        label_col ="score" # For loss_type == rbf: label_col="label"
        
        # Load data
        if loss_type in ['cosine', 'cosent']:
            train_dataset, eval_dataset = load_and_clean_data_cosine(train_path, val_path, approach, kernel_task)            
        else: 
            train_dataset, eval_dataset = load_and_clean_data_rbf(train_path, val_path, approach)
            
        for val in target_values:
            train_dataset = oversample_min_classes(train_dataset, label_col=label_col, target_frac=0.4, target_value=val)
        logger.info("Train DS class districution after:", train_dataset.to_pandas()[label_col].value_counts())

        if loss_type == "cosent": 
            loss = CoSENTLoss(model)
            logger.info("Using CoSENT Loss")
        elif loss_type == "mse": 
            loss = EuclideanDistanceLoss(model)
            logger.info("Using Euc. loss")
        else:
            loss = CosineSimilarityLoss(model)
            logger.info("Using standard CosineSimilarityLoss")
        
        logger.info(f"Using loss function: {loss.__class__.__name__}")
        
        logger.info("Creating evaluator")
        dev_evaluator = CustomEvaluator(
            sentences1=eval_dataset["sentence1"],
            sentences2=eval_dataset["sentence2"],
            scores=eval_dataset["score"],
            name="custom",
        )
        # # -----------------EXPERIMENTAL
        # else:
        #     if loss_type == "cosent": 
        #         loss = DeltaCoSENTLoss(model)
        #         logger.info("Using Cosent loss")
        #     else:
        #         loss = DeltaCosineSimilarityLoss(model)
        #         logger.info("Using DeltaCosineSimilarityLoss")
                
        #     # Create delta-based evaluator
        #     dev_evaluator = DeltaEvaluator(
        #         prefixes=eval_dataset["prefix"],
        #         sentences1=eval_dataset["sentence1"],
        #         sentences2=eval_dataset["sentence2"],
        #         scores=eval_dataset["score"],
        #         name="custom",
        #     )            
            
        logger.info("Evaluating base model")
        base_score = dev_evaluator(model)
        logger.info(f"Base model score: {base_score}")
        
        run_name = f"{model_name_only}-{kernel_task}-peft-weighted_lora_neutral" if use_lora else model_name_only
        args = create_training_args(run_name, num_epochs, batch_size)
        
        logger.info("Initializing trainer")
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=dev_evaluator,
        )
        
        logger.info("Starting training")
        trainer.train()
        
        logger.info("Evaluating trained model")
        final_score = dev_evaluator(model)
        logger.info(f"Base model score: {base_score}")
        logger.info(f"Final model score: {final_score}")
        
        final_output_dir = f"models_peft/{run_name}/final"
        Path(final_output_dir).parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_output_dir)
        logger.info(f"Model saved to {final_output_dir}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()