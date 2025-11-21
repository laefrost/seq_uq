import logging
import sys
import traceback

from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss, CosineSimilarityLoss
from sentence_transformers.training_args import BatchSamplers


from utils.utils import save, load

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# You can specify any Hugging Face pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model_name_only = model_name.split("/")[-1]

# 1. Load a model to finetune with 2. (Optional) model card data
model = SentenceTransformer(
    model_name,
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name=f"{model_name_only} finetuned adapter",
    ),
)

# Create a LoRA adapter for the model
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=64,
    lora_alpha=128,
    lora_dropout=0.1,
)
model.add_adapter(peft_config)


# TODO: Adapt this for my use case
# 3. Load a dataset to finetune on
# dataset = load_dataset("sentence-transformers/gooaq", split="train")
# dataset_dict = dataset.train_test_split(test_size=10_000, seed=12)
# train_dataset: Dataset = dataset_dict["train"].select(range(1_000_000))
# eval_dataset: Dataset = dataset_dict["test"]
# Was brauche ich: sentence1, sentence2, score
train_dict = load('finetuning/train_dict.pkl')
val_dict = load('finetuning/train_dict.pkl')

train_dataset: Dataset = Dataset.from_list(train_dict)#load_dataset("csv", data_files="my_file.csv") #Dataset.from_dict()
eval_dataset: Dataset = Dataset.from_list(val_dict)#load_dataset("csv", data_files="my_file.csv") #Dataset.from_dict()

loss = CosineSimilarityLoss(model)

# 5. (Optional) Specify training arguments
run_name = f"{model_name_only}-peft"
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=25,
    logging_first_step=True,
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
)

# 6. (Optional) Create an evaluator & evaluate the base model
# The full corpus, but only the evaluation queries
dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    scores=eval_dataset["label"],
    name="sts_dev",
)
dev_evaluator(model)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

# (Optional) Evaluate the trained model on the evaluator after training
dev_evaluator(model)

# 8. Save the trained model
final_output_dir = f"models_peft/{run_name}/final"
model.save_pretrained(final_output_dir)

# 9. (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
# try:
#     model.push_to_hub(run_name)
# except Exception:
#     logging.error(
#         f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
#         f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
#         f"and saving it using `model.push_to_hub('{run_name}')`."
#     )