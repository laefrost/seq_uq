"""
FORTGESCHRITTENE VERSION: Fine-tuning mit gewichteter Attention auf letzte Tokens
Diese Version modifiziert das Pooling, um die letzten Tokens stärker zu gewichten
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sentence_transformers.models import Module

import torch
import torch.nn as nn
from transformers import Trainer

class WeightedLossTrainer(Trainer):
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


class WeightedPooling(Module):
    """
    Custom Pooling Layer, der die letzten Tokens stärker gewichtet
    """
    def __init__(self, word_embedding_dimension: int, pooling_mode: str = 'exponential'):
        super(WeightedPooling, self).__init__()
        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode = pooling_mode
        
        # Lernbare Gewichte für Position-basiertes Weighting
        self.position_weights = nn.Parameter(torch.ones(512))  # Max 512 tokens
        
    def forward(self, features):
        """
        features: Dict mit 'token_embeddings' und 'attention_mask'
        """
        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']
        
        # Berechne Sequenzlängen
        seq_lengths = attention_mask.sum(dim=1)
        
        # Erstelle Positionsgewichte
        batch_size, seq_len, hidden_dim = token_embeddings.shape
        
        if self.pooling_mode == 'exponential':
            # Exponentiell steigende Gewichte zum Ende
            positions = torch.arange(seq_len, device=token_embeddings.device).unsqueeze(0)
            positions = positions.expand(batch_size, -1)
            
            # Normalisiere Positionen auf [0, 1] basierend auf tatsächlicher Länge
            normalized_positions = positions.float() / seq_lengths.unsqueeze(1).float()
            
            # Exponentielles Weighting: stärkere Gewichte am Ende
            # weight = exp(alpha * normalized_position)
            alpha = 3.0  # Stärke des Weightings (höher = mehr Focus auf Ende)
            weights = torch.exp(alpha * normalized_positions)
            
        elif self.pooling_mode == 'linear':
            # Linear steigende Gewichte
            positions = torch.arange(seq_len, device=token_embeddings.device).unsqueeze(0)
            positions = positions.expand(batch_size, -1)
            normalized_positions = positions.float() / seq_lengths.unsqueeze(1).float()
            weights = 1.0 + 2.0 * normalized_positions  # Von 1.0 bis 3.0
            
        elif self.pooling_mode == 'last_n':
            # Nur die letzten N Tokens verwenden
            n = 5
            positions = torch.arange(seq_len, device=token_embeddings.device).unsqueeze(0)
            positions = positions.expand(batch_size, -1)
            weights = (positions >= (seq_lengths.unsqueeze(1) - n)).float()
        
        else:  # learned
            # Verwende lernbare Gewichte
            weights = self.position_weights[:seq_len].unsqueeze(0).expand(batch_size, -1)
            weights = torch.softmax(weights, dim=1)
        
        # Anwende Attention Mask
        weights = weights * attention_mask.float()
        
        # Normalisiere Gewichte
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        
        # Gewichtetes Mean Pooling
        weights = weights.unsqueeze(-1).expand_as(token_embeddings)
        weighted_embeddings = token_embeddings * weights
        sentence_embedding = weighted_embeddings.sum(dim=1)
        
        features.update({'sentence_embedding': sentence_embedding})
        return features
    
    def get_sentence_embedding_dimension(self):
        return self.word_embedding_dimension
    
    def save(self, output_path, *args, safe_serialization=True, **kwargs) -> None:
        self.save_config(output_path) 
        


def create_model_with_weighted_pooling(model_name='sentence-transformers/all-MiniLM-L6-v2', 
                                       pooling_mode='exponential'):
    """
    Erstellt ein SentenceTransformer Modell mit custom weighted pooling
    """
    # Lade Word Embedding Model
    word_embedding_model = models.Transformer(model_name)
    
    # Erstelle Custom Pooling Layer
    pooling_model = WeightedPooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode=pooling_mode
    )
    
    # Erstelle Dense Layer (optional, für Dimensionsreduktion)
    dense_model = models.Dense(
        in_features=pooling_model.get_sentence_embedding_dimension(),
        out_features=256,  # Reduziere auf 256 Dimensionen
        activation_function=nn.Tanh()
    )
    
    # Normalization Layer
    normalize_model = models.Normalize()
    
    # Kombiniere alle Module
    model = SentenceTransformer(modules=[
        word_embedding_model,
        pooling_model,
        dense_model,
        normalize_model
    ])
    
    return model