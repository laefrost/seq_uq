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
from sentence_transformers.losses import CosineSimilarityLoss

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


def load_and_clean_data(train_path: str, val_path: str):
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
    required_cols = ['sentence1', 'sentence2', 'label']
    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # CRITICAL: Format data correctly for CosineSimilarityLoss
    # The dataset must have columns: sentence1, sentence2, score (not label)
    # Rename 'label' to 'score' if needed
    if 'label' in train_df.columns and 'score' not in train_df.columns:
        train_df = train_df.rename(columns={'label': 'score'})
        val_df = val_df.rename(columns={'label': 'score'})
    
    # Ensure score is float type
    train_df['score'] = train_df['score'].astype(float)
    val_df['score'] = val_df['score'].astype(float)
    
    # Log sample data to verify format
    logger.info(f"Sample train data:\n{train_df.head(2)}")
    logger.info(f"Score range - Train: [{train_df['score'].min()}, {train_df['score'].max()}]")
    logger.info(f"Score range - Val: [{val_df['score'].min()}, {val_df['score'].max()}]")
    
    # Convert to datasets - keep only required columns
    train_dataset = Dataset.from_pandas(
        train_df[['sentence1', 'sentence2', 'score']], 
        preserve_index=False
    )
    eval_dataset = Dataset.from_pandas(
        val_df[['sentence1', 'sentence2', 'score']], 
        preserve_index=False
    )
    
    logger.info(f"Loaded {len(train_dataset)} training samples and {len(eval_dataset)} validation samples")
    logger.info(f"Train dataset features: {train_dataset.features}")
    
    return train_dataset, eval_dataset


def setup_model(model_name: str, use_lora: bool = True):
    """Initialize the sentence transformer model with optional LoRA adapter."""
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
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        # Logging
        logging_steps=25,
        logging_first_step=True,
        run_name=run_name,
        # Additional useful parameters
        load_best_model_at_end=True,
        metric_for_best_model="eval_sts_dev_spearman_cosine",
        greater_is_better=True,
    )


def main():
    """Main training pipeline."""
    try:
        # Configuration
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        train_path = 'finetuning/train.xlsx'
        val_path = 'finetuning/val.xlsx'
        num_epochs = 150  # Changed from 500 to a more reasonable value
        batch_size = 32
        use_lora = True
        
        # Load data
        train_dataset, eval_dataset = load_and_clean_data(train_path, val_path)
        
        # Setup model
        model, model_name_only = setup_model(model_name, use_lora=use_lora)
        
        # Initialize loss function
        loss = CosineSimilarityLoss(model)
        logger.info(f"Using loss function: {loss.__class__.__name__}")
        
        # Create evaluator - use 'score' column instead of 'label'
        logger.info("Creating evaluator")
        dev_evaluator = EmbeddingSimilarityEvaluator(
            sentences1=eval_dataset["sentence1"],
            sentences2=eval_dataset["sentence2"],
            scores=eval_dataset["score"],  # Changed from 'label' to 'score'
            name="sts_dev",
        )
        
        # Evaluate base model
        logger.info("Evaluating base model")
        base_score = dev_evaluator(model)
        logger.info(f"Base model score: {base_score}")
        
        # Setup training arguments
        run_name = f"{model_name_only}-peft" if use_lora else model_name_only
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

# import logging
# import sys
# import traceback
# import pandas as pd

# from datasets import Dataset, load_dataset
# from peft import LoraConfig, TaskType

# from sentence_transformers import (
#     SentenceTransformer,
#     SentenceTransformerModelCardData,
#     SentenceTransformerTrainer,
#     SentenceTransformerTrainingArguments,
# )
# from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
# from sentence_transformers.losses import CachedMultipleNegativesRankingLoss, CosineSimilarityLoss
# from sentence_transformers.training_args import BatchSamplers


# from utils.utils import save, load

# # Set the log level to INFO to get more information
# logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# # You can specify any Hugging Face pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
# model_name = 'sentence-transformers/all-MiniLM-L6-v2'
# model_name_only = model_name.split("/")[-1]

# # 1. Load a model to finetune with 2. (Optional) model card data
# model = SentenceTransformer(
#     model_name,
#     model_card_data=SentenceTransformerModelCardData(
#         language="en",
#         license="apache-2.0",
#         model_name=f"{model_name_only} finetuned adapter",
#     ),
# )

# # Create a LoRA adapter for the model
# peft_config = LoraConfig(
#     task_type=TaskType.FEATURE_EXTRACTION,
#     inference_mode=False,
#     r=64,
#     lora_alpha=128,
#     lora_dropout=0.1,
# )
# model.add_adapter(peft_config)


# # TODO: Adapt this for my use case
# # 3. Load a dataset to finetune on
# # dataset = load_dataset("sentence-transformers/gooaq", split="train")
# # dataset_dict = dataset.train_test_split(test_size=10_000, seed=12)
# # train_dataset: Dataset = dataset_dict["train"].select(range(1_000_000))
# # eval_dataset: Dataset = dataset_dict["test"]
# # Was brauche ich: sentence1, sentence2, score
# train_df = pd.read_excel('finetuning/train.xlsx')
# val_df = pd.read_excel('finetuning/val.xlsx')

# # Clean NaN values
# for df in [train_df, val_df]:
#     for col in df.columns:
#         if df[col].dtype == 'object':
#             df[col] = df[col].fillna('').astype(str)
#         else:
#             df[col] = df[col].fillna(0)

# train_dict= train_df.to_dict('records') 
# val_dict= val_df.to_dict('records') 

# # # Create datasets from cleaned DataFrames
# # train_dataset = Dataset.from_pandas(train_df)
# # eval_dataset = Dataset.from_pandas(val_df)

# train_dataset: Dataset = Dataset.from_list(train_dict)#load_dataset("csv", data_files="my_file.csv") #Dataset.from_dict()
# eval_dataset: Dataset = Dataset.from_list(val_dict)#load_dataset("csv", data_files="my_file.csv") #Dataset.from_dict()

# loss = CosineSimilarityLoss(model)

# # 5. (Optional) Specify training arguments
# run_name = f"{model_name_only}-peft"
# args = SentenceTransformerTrainingArguments(
#     # Required parameter:
#     output_dir=f"models/{run_name}",
#     # Optional training parameters:
#     num_train_epochs=500,
#     report_to = "none", 
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=32,
#     learning_rate=2e-5,
#     warmup_ratio=0.1,
#     fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
#     bf16=True,  # Set to True if you have a GPU that supports BF16
#     # Optional tracking/debugging parameters:
#     eval_strategy="steps",
#     eval_steps=100,
#     save_strategy="steps",
#     save_steps=100,
#     save_total_limit=2,
#     logging_steps=25,
#     logging_first_step=True,
#     run_name=run_name,  # Will be used in W&B if `wandb` is installed
# )

# # 6. (Optional) Create an evaluator & evaluate the base model
# # The full corpus, but only the evaluation queries
# dev_evaluator = EmbeddingSimilarityEvaluator(
#     sentences1=eval_dataset["sentence1"],
#     sentences2=eval_dataset["sentence2"],
#     scores=eval_dataset["label"],
#     name="sts_dev",
# )
# dev_evaluator(model)

# # 7. Create a trainer & train
# trainer = SentenceTransformerTrainer(
#     model=model,
#     args=args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     loss=loss,
#     evaluator=dev_evaluator,
# )
# trainer.train()

# # (Optional) Evaluate the trained model on the evaluator after training
# dev_evaluator(model)

# # 8. Save the trained model
# final_output_dir = f"models_peft/{run_name}/final"
# model.save_pretrained(final_output_dir)

# # 9. (Optional) save the model to the Hugging Face Hub!
# # It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
# # try:
# #     model.push_to_hub(run_name)
# # except Exception:
# #     logging.error(
# #         f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
# #         f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
# #         f"and saving it using `model.push_to_hub('{run_name}')`."
# #     )