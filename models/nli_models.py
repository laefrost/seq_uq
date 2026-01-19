import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed, pipeline
from huggingface_hub import notebook_login, login
import numpy as np
from huggingface_hub import InferenceClient
import os, gc
import re
# import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class NLI(): 
    def __init__(self, model_id = 'microsoft/deberta-v3-base-mnli'): 
        self.model_id = model_id       
        self.pipe = pipeline(
            "text-classification",
            model=self.model_id,
            device=0,              # GPU if available
            torch_dtype="auto"
        )

    def check_implication_batch(self, batch_pairs, question = None, mode="data"):
        inputs = [
            {'text': t1, 
             'text_pair' : t2}
            for (t1, t2) in batch_pairs
        ]
            
        outputs = self.pipe(
                inputs
        )
        
        scores = []
        labels = []
        for result in outputs:
            print("in nli ----------------------------------")
            print(result)
            # result_scores = {item["label"].lower(): item["score"] for item in result}
            # print(result_scores)
            # max_score = max(result_scores, key=result_scores.get)
            label = result['label'].lower()
            score = result['score']
            if label == "entailment": 
                value = 2
            elif label == "contradiction": 
                value = 0
            else: 
                value = 1
            labels.append(value)
            scores.append(score)
    
        return labels
    
    