"""Sample answers from LLMs on QA task."""
import gc
import logging
import random
import torch

from models.models import * 
from models.nli_models import NLI
from uncertainty_metrics.se import * 
from utils.subsequences import generate_subsequences
from utils.utils import get_parser, construct_prompt, save, setup_logger, load
from data.utils import load_ds

setup_logger()

def main(args):
    """
    Generates LLM answers for a QA dataset and produces pairwise NLI-labeled
    training data from sampled token subsequences.

    The pipeline runs in two stages:
    1. Answer generation: for each dataset example, the LLM generates a response
       and extracts per-step token subsequences via sampling.
    2. NLI labeling: for each generation step, all unique decoded subsequences
       are paired and classified (entailment / neutral / contradiction) using an
       NLI model. The resulting labeled pairs are saved as training data.

    Intermediate generation results and final labeled entries are saved to disk
    using the experiment name and dataset name as filename prefixes.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments. Expected fields:
        - random_seed (int): Seed for dataset sampling and reproducibility.
        - num_samples (int): Number of dataset examples to process.
        - dataset (str): Dataset identifier passed to load_ds.
        - ellm_model_id (str): HuggingFace model ID for the NLI entailment model.
        - exp_name (str): Experiment name prefix used in output filenames.
        - task_type (str): Task type passed to construct_prompt.
        - k (int): Reserved for top-k sampling configuration.
    """
    experiment_details = {'args': args}
    random.seed(args.random_seed)

    samples = load_ds(args.dataset, seed=args.random_seed, num_samples = args.num_samples)
    
    logging.info('Dataset loaded!')
    
    model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    ellm_model_id = args.ellm_model_id
    exp_name = args.exp_name
    ds_name = args.dataset
    task_type = args.task_type
    k = args.k
    
    # # Initialize model
    llm = LLM(model_id=model_id)
    logging.info('Model init')

    # Start answer generation.
    logging.info(80 * '=')
    logging.info('Generating answers: ')
    logging.info(80 * '=')
        
    generations = []

    it = 0
    for s, example in enumerate(samples): 
        if (it + 1 % 10) == 0:
            gc.collect()
            torch.cuda.empty_cache()
        it += 1
        
        prompt = construct_prompt(example['question'], task_type=task_type)
        
        generated_text, step_sequences, gen_ids, gen_tokens = llm.generate_with_topk(prompt=prompt, temperature = 0.9)
        seq_tokens_sampled = generate_subsequences(step_sequences, llm.tokenizer, gen_ids, sampling_k = k, scaling_p = None, selection_p = 0.95, method = "sampling", question = prompt)
        
        generations.append({
            'example' : example,
            'generated_text' : generated_text, 
            'seq_tokens_sampled' : seq_tokens_sampled}
        )
        
    del llm
    
    save(generations, f'{exp_name}_{ds_name}_data_generations.pkl')
    save(experiment_details, f'{exp_name}_{ds_name}_data_details.pkl')
    logging.info('Run complete.')
    
    generations = load(f'{exp_name}_{ds_name}_data_generations.pkl')
    
    entries = []

    ellm = NLI(model_id=ellm_model_id)

    MAX_BATCH = 32
    for e, element in enumerate(generations): 
        logging.info(element['generated_text'])
        example = element['example']
        question = example['question']
        # seq_tokens = element['seq_tokens']
        sampled_tokens = element['seq_tokens_sampled']
                            
        for s, step in enumerate(sampled_tokens): 
            decoded_seqs = step.get('alternative_sequence_decoded', None) 
            decoded_seqs = list(set(decoded_seqs))
            prefix = step.get('prev_seq_decoded', '')#llm.tokenizer.decode(step['prev_seq'], skip_special_tokens = True)
            
            if len(decoded_seqs) == 1: 
                continue
                
            checked_ids = []   
            batched_pairs = []            
            for i, string1 in enumerate(decoded_seqs):
                for j in range(1, len(decoded_seqs)):
                    string2 = decoded_seqs[j]
                    if (i, j) in checked_ids: 
                        continue
                    if i == j or string1 == string2: 
                        continue
                    batched_pairs.append((string1, string2))
                    checked_ids.append((i, j))
                    checked_ids.append((j, i))
                    
            all_labels = []
            for b in range(0, len(batched_pairs), MAX_BATCH):
                sub = batched_pairs[b:b+MAX_BATCH]
                labels = ellm.check_implication_batch(sub)
                all_labels.extend(labels)
            logging.info(all_labels)    
            for (p1, p2), label in zip(batched_pairs, all_labels):
                entry = {
                    'generated_text': element['generated_text'],
                    'true_answer': element['example']['answer'],
                    'question': question,
                    'index': s,
                    'prefix': prefix,
                    'text1': p1,
                    'text2': p2,
                    'label' : label
                }
                entries.append(entry)
    save(entries, f'{exp_name}_{ds_name}_data.pkl')
    del ellm
    
    
if __name__ == '__main__':
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    logging.info('STARTING `generate_training_data`!')
    main(args)
    logging.info('FINISHED `generate_training_data`!')