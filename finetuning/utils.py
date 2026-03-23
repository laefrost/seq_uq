import logging
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
from peft import LoraConfig, TaskType
from sentence_transformers import (
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch import nn, Tensor
from typing import Iterable, Dict

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import accuracy_score, f1_score, classification_report

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('delta_training.log')
    ]
)

logger = logging.getLogger(__name__)

class WeightedLossTrainer(Trainer):
    """
    Replaces the standard cross-entropyloss with a class-weighted variant.
    If no `class_weights` are provided at construction time, a default
    weight tensor of [3.0, 1.0, 1.0] is used (upweighting class 0).

    Args:
        class_weights (torch.Tensor | None): 1-D float tensor of per-class
            weights with length equal to the number of labels.
    """
    def __init__(self, model = None, args = None, data_collator = None, train_dataset = None, eval_dataset = None, processing_class = None, model_init = None, compute_loss_func = None, compute_metrics = None, callbacks = None, optimizers = (None, None), optimizer_cls_and_kwargs = None, preprocess_logits_for_metrics = None, class_weights = None ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, processing_class, model_init, compute_loss_func, compute_metrics, callbacks, optimizers, optimizer_cls_and_kwargs, preprocess_logits_for_metrics)
        self.class_weights = class_weights 
        
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is None: 
            class_weights = torch.tensor([3.0, 1.0, 1.0]).to(logits.device)
        else:
            class_weights = self.class_weights.to(logits.device)
        
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss
    

class EuclideanDistanceLoss(nn.Module):
    """
    MSE loss over Euclidean distances between pairs of sentence embeddings.

    Encodes both sentences in a pair, computes their L2 distance (with a
    small epsilon for numerical stability), and minimizes the mean squared
    error between those distances and target values derived from the labels.

    When `similarity_to_distance` is True (default), labels are assumed to
    be similarity scores in [0, 1] and are converted to distance targets as
    `target = 1 - label`. When False, labels are treated directly as target
    distances.

    Args:
        model (SentenceTransformer): The sentence encoder to optimize.
        similarity_to_distance (bool): Whether to invert similarity labels
            into distance targets. Default True.
    """
    def __init__(self, model: SentenceTransformer, similarity_to_distance: bool = True):
        super(EuclideanDistanceLoss, self).__init__()
        self.model = model
        self.similarity_to_distance = similarity_to_distance
        self.mse_loss = nn.MSELoss()
    
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # Get embeddings
        embeddings_a = self.model(sentence_features[0])['sentence_embedding']
        embeddings_b = self.model(sentence_features[1])['sentence_embedding']
        
        # Compute Euclidean distance
        euclidean_dist = torch.sqrt(torch.sum((embeddings_a - embeddings_b) ** 2, dim=1) + 1e-8)
        
        # Convert similarity to distance if needed
        if self.similarity_to_distance:
            # Assuming labels are similarities [0, 1], convert to distances
            target_distances = 1.0 - labels.view(-1)
        else:
            target_distances = labels.view(-1)
        
        return self.mse_loss(euclidean_dist, target_distances)
 
 
# # -----------------EXPERIMENTAL
# class DeltaCoSENTLoss(nn.Module):
#     def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct=util.pairwise_cos_sim) -> None:
#         super().__init__()
#         self.model = model
#         self.similarity_fct = similarity_fct
#         self.scale = scale

#     def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
#         prefix_emb = self.model(sentence_features[0])['sentence_embedding']
#         subseq1_emb = self.model(sentence_features[1])['sentence_embedding']
#         subseq2_emb = self.model(sentence_features[2])['sentence_embedding']
#         prefix_emb_detached = prefix_emb.detach()
        
#         # Compute deltas using detached prefix
#         delta1 = subseq1_emb - prefix_emb_detached
#         delta2 = subseq2_emb - prefix_emb_detached

#         return self.compute_loss_from_embeddings([delta1, delta2], labels)

#     def compute_loss_from_embeddings(self, embeddings: list[Tensor], labels: Tensor) -> Tensor:
#         scores = self.similarity_fct(embeddings[0], embeddings[1])
#         scores = scores * self.scale
#         scores = scores[:, None] - scores[None, :]

#         # label matrix indicating which pairs are relevant
#         labels = labels[:, None] < labels[None, :]
#         labels = labels.float()

#         # mask out irrelevant pairs so they are negligible after exp()
#         scores = scores - (1 - labels) * 1e12

#         # append a zero as e^0 = 1
#         scores = torch.cat((torch.zeros(1).to(scores.device), scores.view(-1)), dim=0)
#         loss = torch.logsumexp(scores, dim=0)

#         return loss

    
# class DeltaCosineSimilarityLoss(nn.Module):
#     """
#     Custom loss that computes deltas before computing cosine similarity.
    
#     Input format: Each row contains [prefix, subseq1, subseq2, label]
    
#     Computes:
#         d1 = embed(subseq1) - embed(prefix)
#         d2 = embed(subseq2) - embed(prefix)
#         loss = MSE(cos(d1, d2), label)
#     """
    
#     def __init__(self, model, contradiction_weight=100.0, neutral_weight=0.5):
#         super().__init__()
#         self.model = model
#         self.contradiction_weight = contradiction_weight
#         self.neutral_weight = neutral_weight
#         self.loss_fct = nn.MSELoss(reduction='none')
        
#     def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
#         """
#         sentence_features contains 3 elements: [prefix, subseq1, subseq2]
#         """
#         # Get embeddings
#         prefix_emb = self.model(sentence_features[0])['sentence_embedding']
#         subseq1_emb = self.model(sentence_features[1])['sentence_embedding']
#         subseq2_emb = self.model(sentence_features[2])['sentence_embedding']
        
#         # CRITICAL: Detach prefix embedding so gradients don't flow through it
#         # We want to learn how to embed subsequences, not how to change the prefix
#         prefix_emb_detached = prefix_emb.detach()
        
#         # Compute deltas using detached prefix
#         delta1 = subseq1_emb - prefix_emb_detached
#         delta2 = subseq2_emb - prefix_emb_detached
        
#         # Compute cosine similarity on deltas
#         output = torch.cosine_similarity(delta1, delta2)        
#         # Compute per-sample loss
#         loss_per_sample = self.loss_fct(output, labels.view(-1))
        
#         # Apply weights based on label
#         weights = torch.ones_like(labels)
#         weights[labels == -1] = self.contradiction_weight  # High weight for contradictions
#         weights[labels == 0] = self.neutral_weight  # Lower weight for neutrals
#         weights[labels == 1] = 5  # Normal weight for entailments
        
#         # Weighted loss
#         weighted_loss = (loss_per_sample * weights).mean()
        
#         # Compute loss
#         # loss = self.loss_fct(output, labels.view(-1))
#         return weighted_loss
    
    
# class CustomEvaluator(EmbeddingSimilarityEvaluator):
#     def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
#         # Get base metrics
#         base_score = super().__call__(model, output_path, epoch, steps)
        
#         # Compute predictions
#         embeddings1 = model.encode(self.sentences1, convert_to_numpy=True)
#         embeddings2 = model.encode(self.sentences2, convert_to_numpy=True)
        
#         cosine_scores = []
#         for emb1, emb2 in zip(embeddings1, embeddings2):
#             cosine_scores.append(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        
#         cosine_scores = np.array(cosine_scores)
#         true_labels = np.array(self.scores)
        
#         # Prediction bins
#         predicted_labels = np.zeros_like(cosine_scores)
#         predicted_labels[cosine_scores > 0.5] = 1.0    # Entailment
#         predicted_labels[cosine_scores < -0.5] = -1.0  # Contradiction
        
#         # Per-class metrics
#         for label_val, label_name in [(-1, "Contradiction"), (0, "Neutral"), (1, "Entailment")]:
#             mask = true_labels == label_val
#             if mask.sum() > 0:
#                 accuracy = (predicted_labels[mask] == label_val).mean()
#                 avg_score = cosine_scores[mask].mean()
#                 logger.info(f"{label_name} - Accuracy: {accuracy:.3f}, Avg Cosine: {avg_score:.3f}, Count: {mask.sum()}")
        
#         # Overall accuracy
#         overall_accuracy = (predicted_labels == true_labels).mean()
#         logger.info(f"Overall Accuracy: {overall_accuracy:.3f}")
        
#         return base_score
    
    
# class DeltaEvaluator(EmbeddingSimilarityEvaluator):
#     """
#     Custom evaluator that uses delta embeddings for evaluation.
#     """
    
#     def __init__(self, prefixes, sentences1, sentences2, scores, name=""):
#         # Store prefixes
#         self.prefixes = prefixes
        
#         # Call parent init
#         super().__init__(
#             sentences1=sentences1,
#             sentences2=sentences2,
#             scores=scores,
#             name=name
#         )
        
#     def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
#         """Evaluate using delta embeddings."""
        
#         # Encode all texts
#         prefix_embs = model.encode(self.prefixes, convert_to_numpy=True, show_progress_bar=False)
#         sent1_embs = model.encode(self.sentences1, convert_to_numpy=True, show_progress_bar=False)
#         sent2_embs = model.encode(self.sentences2, convert_to_numpy=True, show_progress_bar=False)
        
#         # Compute deltas
#         delta1 = sent1_embs - prefix_embs
#         delta2 = sent2_embs - prefix_embs
        
#         # Compute cosine similarities on deltas
#         cosine_scores = []
#         for d1, d2 in zip(delta1, delta2):
#             sim = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-8)
#             cosine_scores.append(sim)
        
#         cosine_scores = np.array(cosine_scores)
#         true_labels = np.array(self.scores)
        
#         # Compute metrics
#         from scipy.stats import spearmanr, pearsonr
        
#         spearman = spearmanr(true_labels, cosine_scores)[0]
#         pearson = pearsonr(true_labels, cosine_scores)[0]
        
#         # Per-class accuracy
#         predicted_labels = np.zeros_like(cosine_scores)
#         predicted_labels[cosine_scores > 0.5] = 1.0
#         predicted_labels[cosine_scores < -0.5] = -1.0
        
#         logger.info(f"\n{self.name} - Delta-based Evaluation:")
#         logger.info(f"  Spearman: {spearman:.4f}")
#         logger.info(f"  Pearson:  {pearson:.4f}")
        
#         for label_val, label_name in [(-1, "Contradiction"), (0, "Neutral"), (1, "Entailment")]:
#             mask = true_labels == label_val
#             if mask.sum() > 0:
#                 accuracy = (predicted_labels[mask] == label_val).mean()
#                 avg_score = cosine_scores[mask].mean()
#                 logger.info(f"  {label_name:13} - Acc: {accuracy:.3f}, Avg: {avg_score:.3f}, N: {mask.sum()}")
        
#         overall_acc = (predicted_labels == true_labels).mean()
#         logger.info(f"  Overall Accuracy: {overall_acc:.3f}")
        
#         return pearson