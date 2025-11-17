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
            {'question' : 'Who is John Russel Reynolds?', 'answer' : None}, 
            {'question' : 'Who is Adja Yunkers?', 'answer' : None},
            {'question' : 'Who is Bert Deacon?', 'answer' : None},
            {'question' : "Who is Freddie Frith?", 'answer' : None}, 
            # {'question' : 'Who is Wilhelm Windelband?'},
            # {'question' : "Who is Moisés Kaufman?"},
            # {'question' : 'Who is Laurent Koscielny?'},
            # {'question' : "Who is Véra Korène?"},
            # {'question' : 'Who is Leana de Bruin?'},
            # {'question' : 'Who is Tera Van Beilen?}'}
        ]
    
    return sampled_examples
   
    