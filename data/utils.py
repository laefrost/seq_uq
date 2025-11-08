from datasets import load_dataset
import random

def load_ds(dataset_name, seed, add_options=None, num_samples = 5): 
    random.seed(seed)
    if dataset_name == 'trivia_qa':
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        sampled_examples = random.sample(list(dataset), num_samples)
    
    return sampled_examples