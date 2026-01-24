from datasets import load_dataset
import random

def load_ds(dataset_name, seed, add_options=None, num_samples = 5): 
    random.seed(seed)
    if dataset_name == 'trivia_qa':
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        sampled_examples = random.sample(list(dataset), num_samples)
    if dataset_name == 'trivia_qa_data':
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
        sampled_examples = random.sample(list(dataset), num_samples)
    

    elif dataset_name == 'factual_bio':
        sampled_examples = [
            {'question' : 'Who is John Russel Reynolds?', 'answer' : None, 'topic': 'John Russel Reynolds'}, 
            {'question' : 'Who is Adja Yunkers?', 'answer' : None, 'topic': 'Adja Yunkers'},
            {'question' : 'Who is Bert Deacon?', 'answer' : None, 'topic': 'Bert Deacon'},
            {'question' : "Who is Freddie Frith?", 'answer' : None, 'topic': 'Freddie Frith'}, 
            # {'question' : 'Who is Wilhelm Windelband?'},
            # {'question' : "Who is Moisés Kaufman?"},
            # {'question' : 'Who is Laurent Koscielny?'},
            # {'question' : "Who is Véra Korène?"},
            # {'question' : 'Who is Leana de Bruin?'},
            # {'question' : 'Who is Tera Van Beilen?}'}
        ]
        
    elif dataset_name == 'factscore_bio': 
        with open("data/prompt_entities.txt", "r", encoding="utf-8") as f:
            entities = f.read().splitlines()
        entities_reduced = random.sample(list(entities), num_samples)
        sampled_examples = []    
        for ent in entities_reduced: 
            sampled_examples.append({'question' : f'Tell me a bio of {ent}.', 'answer' : None, 'topic': str(ent)})
            
    return sampled_examples
   
    