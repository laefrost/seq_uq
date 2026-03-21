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
from finetuning.models import create_model_with_weighted_pooling
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

def load_and_clean_data_rbf(train_path: str, val_path: str, approach = 'og'):
    """Load and clean training and validation data from Excel files."""
    logger.info(f"Loading data from {train_path} and {val_path}")
    
    train_df = pd.read_excel(train_path)
    val_df = pd.read_excel(val_path)
    
    # Log data info before cleaning
    logger.info(f"Train data shape: {train_df.shape}, Val data shape: {val_df.shape}")
    logger.info(f"Train columns: {train_df.columns.tolist()}")
    
    # Clean NaN values
    for df in [train_df, val_df]:
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('').astype(str)
            else:
                df[col] = df[col].fillna(0)
    
    # Validate required columns
    # required_cols = ['sentence1', 'sentence2', 'label']
    #for col in required_cols:
    #    if col not in train_df.columns:
    #        raise ValueError(f"Missing required column: {col}")
    
    # CRITICAL: Format data correctly for CosineSimilarityLoss
    # The dataset must have columns: sentence1, sentence2, score (not label)
    # Rename 'label' to 'score' if needed
    if 'score' in train_df.columns and 'label' not in train_df.columns:
        train_df = train_df.rename(columns={'score': 'label'})        
        
    if 'score' in val_df.columns and 'label' not in val_df.columns:
        val_df = val_df.rename(columns={'score': 'label'})
    
    # Ensure score is float type
    train_df['label'] = train_df['label'].astype(float)
    val_df['label'] = val_df['label'].astype(float)
    
    # Log sample data to verify format
    logger.info(f"Sample train data:\n{train_df.head(2)}")
    logger.info(f"Score range - Train: [{train_df['label'].min()}, {train_df['label'].max()}]")
    logger.info(f"Score range - Val: [{val_df['label'].min()}, {val_df['label'].max()}]")
    
    # Convert to datasets - keep only required columns
    if approach == "emb": 
        train_dataset = Dataset.from_pandas(
            train_df[['sentence1', 'sentence2', 'label']], 
            preserve_index=False
        )
        eval_dataset = Dataset.from_pandas(
            val_df[['sentence1', 'sentence2', 'label']], 
            preserve_index=False
        )
    else: 
        train_dataset = Dataset.from_pandas(
            train_df[['prefix', 'sentence1', 'sentence2', 'label']],
            preserve_index=False
        )
        eval_dataset = Dataset.from_pandas(
            val_df[['prefix', 'sentence1', 'sentence2', 'label']], 
            preserve_index=False
        )
    
    logger.info(f"Loaded {len(train_dataset)} training samples and {len(eval_dataset)} validation samples")
    logger.info(f"Train dataset features: {train_dataset.features}")
    
    return train_dataset, eval_dataset

from datasets import concatenate_datasets

def oversample_min_classes(train_dataset, label_col="label", target_value = -1, target_frac=0.2, seed=42):
    # split
    ds_c = train_dataset.filter(lambda x: x[label_col] == target_value)
    ds_o = train_dataset.filter(lambda x: x[label_col] != target_value)

    n_c = len(ds_c)
    n_o = len(ds_o)
    if n_c == 0:
        return train_dataset.shuffle(seed=seed)

    # Wie oft müssen wir ds_c duplizieren, um target_frac zu erreichen?
    # target = (k*n_c) / (k*n_c + n_o)
    # => k = target*n_o / (n_c*(1-target))
    k = int((target_frac * n_o) / (n_c * (1 - target_frac)))
    k = max(1, k)

    ds_c_oversampled = concatenate_datasets([ds_c] * k)
    ds_new = concatenate_datasets([ds_o, ds_c_oversampled]).shuffle(seed=seed)
    return ds_new


def load_and_clean_data_cosine(train_path: str, val_path: str, approach = 'og', kernel_task = 'og'):
    """Load and clean training and validation data from Excel files."""
    logger.info(f"Loading data from {train_path} and {val_path}")
    
    train_df = pd.read_excel(train_path)
    val_df = pd.read_excel(val_path)
    
    # Log data info before cleaning
    logger.info(f"Train data shape: {train_df.shape}, Val data shape: {val_df.shape}")
    logger.info(f"Train columns: {train_df.columns.tolist()}")
    
    # Clean NaN values
    for df in [train_df, val_df]:
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('').astype(str)
            else:
                df[col] = df[col].fillna(0)
    
    # Validate required columns
    # required_cols = ['sentence1', 'sentence2', 'label']
    #for col in required_cols:
    #    if col not in train_df.columns:
    #        raise ValueError(f"Missing required column: {col}")
    
    # CRITICAL: Format data correctly for CosineSimilarityLoss
    # The dataset must have columns: sentence1, sentence2, score (not label)
    # Rename 'label' to 'score' if needed
    if 'label' in train_df.columns and 'score' not in train_df.columns:
        train_df = train_df.rename(columns={'label': 'score'})        
        
    if 'label' in val_df.columns and 'score' not in val_df.columns:
        val_df = val_df.rename(columns={'label': 'score'})
    
    # Ensure score is float type
    train_df['score'] = train_df['score'].astype(float)
    val_df['score'] = val_df['score'].astype(float)
    
    if kernel_task == "claim": 
        train_df['score'] = np.where(train_df['score'].isin([1, -1]), 1, -1)
        val_df['score'] = np.where(val_df['score'].isin([1, -1]), 1, -1)
    elif kernel_task == "cluster": 
        train_df = train_df.loc[train_df["score"] != 0]
        val_df = val_df.loc[val_df["score"] != 0]  
    elif kernel_task == "dispersion":
        train_df['score'] = np.where(train_df['score'].isin([0, -1]), 0, 1)
        val_df['score'] = np.where(val_df['score'].isin([0, -1]), 0, 1)
    elif kernel_task == "contradiction": 
        train_df['score'] == np.select([
            train_df['score'] == -1, 
            train_df['score'] == 0,],
            [0, np.sqrt(2) / 2], default=1)
        
        val_df['score'] == np.select([
            val_df['score'] == -1, 
            val_df['score'] == 0,],
            [0, np.sqrt(2) / 2], default=1)
       
    
    # Log sample data to verify format
    logger.info(f"Sample train data:\n{train_df.head(2)}")
    logger.info(f"Score range - Train: [{train_df['score'].min()}, {train_df['score'].max()}]")
    logger.info(f"Score range - Val: [{val_df['score'].min()}, {val_df['score'].max()}]")
    
    # Convert to datasets - keep only required columns
    if approach == "emb": 
        train_dataset = Dataset.from_pandas(
            train_df[['sentence1', 'sentence2', 'score']], 
            preserve_index=False
        )
        eval_dataset = Dataset.from_pandas(
            val_df[['sentence1', 'sentence2', 'score']], 
            preserve_index=False
        )
    else: 
        train_dataset = Dataset.from_pandas(
            train_df[['prefix', 'sentence1', 'sentence2', 'score']],
            preserve_index=False
        )
        eval_dataset = Dataset.from_pandas(
            val_df[['prefix', 'sentence1', 'sentence2', 'score']], 
            preserve_index=False
        )
    
    logger.info(f"Loaded {len(train_dataset)} training samples and {len(eval_dataset)} validation samples")
    logger.info(f"Train dataset features: {train_dataset.features}")
    
    return train_dataset, eval_dataset


def setup_model(model_name: str, use_lora: bool = True, use_weighted_approach = False, pooling_mode = 'exponential'):
    """Initialize the sentence transformer model with optional LoRA adapter."""
    logger.info(f"Loading model: {model_name}")
    
    model_name_only = model_name.split("/")[-1]
    
    if use_weighted_approach: 
        logger.info(f"Using weighted pooling approach (mode: {pooling_mode})")
        model = create_model_with_weighted_pooling(model_name = model_name, pooling_mode = pooling_mode)
    else: 
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
            #target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            target_modules=["query", "key", "value"],  # Specify target modules
        )
        model.add_adapter(peft_config)
        logger.info(f"LoRA adapter added. Trainable parameters: {model._parameters}")
    
    return model, model_name_only


def create_training_args(run_name: str, num_epochs: int = 10, batch_size: int = 32):
    """Create training arguments with sensible defaults."""
    return SentenceTransformerTrainingArguments(
        output_dir=f"models/{run_name}",
        num_train_epochs=num_epochs,
        report_to="none",  # Change to "wandb" if you want W&B logging
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
        metric_for_best_model='eval_loss', # eval_custom_pearson_cosine 
        greater_is_better=False
    )


def main():
    """Main training pipeline."""
    try:
        # Configuration
        model_name = 'sentence-transformers/all-MiniLM-L6-v2' #'Qwen/Qwen3-Embedding-0.6B' #
        #train_path = 'finetuning/prefix_train_data_pt2.xlsx'
        #val_path = 'finetuning/prefix_val_data.xlsx'
        train_path = 'finetuning/train_final.xlsx'
        val_path = 'finetuning/val.xlsx'
        
        
        num_epochs = 3
        batch_size = 32
        use_lora = True
        loss_type = "cosent" 
        approach = 'emb'
        objective = "cosine"
        kernel_task = "contradiction"
        use_weighted_approach = False
        target_value = 1
        
        # Load data
        if objective == 'cosine':
            train_dataset, eval_dataset = load_and_clean_data_cosine(train_path, val_path, approach, kernel_task)
            logger.info("Train DS class districution before:", train_dataset.to_pandas()["score"].value_counts())
            train_dataset = oversample_min_classes(train_dataset, label_col="score", target_frac=0.4, target_value=0)
            logger.info("Train DS class districution after:", train_dataset.to_pandas()["score"].value_counts())
            train_dataset = oversample_min_classes(train_dataset, label_col="score", target_frac=0.4, target_value=1)
            logger.info("Train DS class districution after:", train_dataset.to_pandas()["score"].value_counts())
        elif objective == 'rbf': 
            train_dataset, eval_dataset = load_and_clean_data_rbf(train_path, val_path, approach)
            logger.info("Train DS class districution before:", Counter(train_dataset["label"]))
            train_dataset = oversample_min_classes(train_dataset, label_col="label", target_frac=0.4, target_value=target_value)
            logger.info("Train DS class districution after:", Counter(train_dataset["label"]))

        # Setup model
        model, model_name_only = setup_model(model_name, use_lora=use_lora, use_weighted_approach = use_weighted_approach)
        
        if approach == "emb": 
            if loss_type == "weighted":
                loss = WeightedCosineSimilarityLoss(
                    model, 
                    contradiction_weight=100.0,  # Increase to focus more on contradictions
                    neutral_weight=0.5
                )
                logger.info("Using WeightedCosineSimilarityLoss")
            elif loss_type == "cosent": 
                loss = CoSENTLoss(model)
                logger.info("Using Cosent loss")
            elif loss_type == "mse": 
                loss = EuclideanDistanceLoss(model)
                logger.info("Using Euc. loss")
            else:
                loss = CosineSimilarityLoss(model)
                logger.info("Using standard CosineSimilarityLoss")
            
            logger.info(f"Using loss function: {loss.__class__.__name__}")
            
            # Create evaluator - use 'score' column instead of 'label'
            logger.info("Creating evaluator")
            # dev_evaluator = EmbeddingSimilarityEvaluator(
            #     sentences1=eval_dataset["sentence1"],
            #     sentences2=eval_dataset["sentence2"],
            #     scores=eval_dataset["score"],  # Changed from 'label' to 'score'
            #     name="sts_dev",
            # )
            
            dev_evaluator = CustomEvaluator(
                sentences1=eval_dataset["sentence1"],
                sentences2=eval_dataset["sentence2"],
                scores=eval_dataset["score"],
                name="custom",
            )
            
            # dev_evaluator = EmbeddingSimilarityEvaluator(
            #     sentences1=eval_dataset["sentence1"],
            #     sentences2=eval_dataset["sentence2"],
            #     scores=eval_dataset["label"],
            #     name="sts_dev",
            # )
        else:
            if loss_type == "cosent": 
                loss = DeltaCoSENTLoss(model)
                logger.info("Using Cosent loss")
            else:
                loss = DeltaCosineSimilarityLoss(model)
                logger.info("Using DeltaCosineSimilarityLoss")
                
            # Create delta-based evaluator
            dev_evaluator = DeltaEvaluator(
                prefixes=eval_dataset["prefix"],
                sentences1=eval_dataset["sentence1"],
                sentences2=eval_dataset["sentence2"],
                scores=eval_dataset["score"],
                name="custom",
            )            
            
        # Evaluate base model
        logger.info("Evaluating base model")
        base_score = dev_evaluator(model)
        logger.info(f"Base model score: {base_score}")
        
        # Setup training arguments
        run_name = f"{model_name_only}-{kernel_task}-peft-weighted_lora_neutral" if use_lora else model_name_only
        args = create_training_args(run_name, num_epochs, batch_size)
        
        # Create trainer
        logger.info("Initializing trainer")
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=dev_evaluator,
        )
        
        # Train
        logger.info("Starting training")
        trainer.train()
        
        # Final evaluation
        logger.info("Evaluating trained model")
        final_score = dev_evaluator(model)
        logger.info(f"Base model score: {base_score}")
        logger.info(f"Final model score: {final_score}")
        
        # Save model
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