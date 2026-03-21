import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed, pipeline
from huggingface_hub import notebook_login, login
import numpy as np
from huggingface_hub import InferenceClient
import os, gc
import re
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig, set_seed, pipeline

load_dotenv()

class NLI(): 
    def __init__(self, model_id='microsoft/deberta-v3-base-mnli'): 
        self.model_id = model_id    
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.eval()   
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.truncation_side = "left" 
        
        self.pipe = pipeline(
            "text-classification",
            model=model,        
            tokenizer=tokenizer,
            device=0,
            torch_dtype="auto"
        )

    def check_implication_batch(self, batch_pairs, question = None, mode="data"):
        inputs = [
            {'text': t1, 
             'text_pair' : t2}
            for (t1, t2) in batch_pairs
        ]
            
        outputs = self.pipe(
                inputs, top_k = None, truncation= "longest_first"
        )
        
        scores = []
        labels = []
        
        for result in outputs:
            score, label = 0, ''
            contr_score = 0
            for label_score in result:
                if label_score['label'].lower() == 'contradiction':
                    contr_score = label_score['score']
                if score < label_score['score']: 
                    label = label_score['label'].lower()
                    score = label_score['score']
                    
            if label == "entailment": 
                value = 2
            elif label == "contradiction": 
                value = 0
            else: 
                value = 1
            labels.append(value)
            scores.append(contr_score)
    
        return labels, scores
    
    