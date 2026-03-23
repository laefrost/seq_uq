"""Sample answers from LLMs on QA task."""
import gc
import logging
import random
import torch

from models.models import * 
from utils.subsequences import generate_words
from utils.utils import get_parser, construct_prompt, save, setup_logger
from data.utils import load_ds

setup_logger()

def main(args):
    """
    Generates LLM answers for a dataset and saves per-example generation data.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments. Expected fields:
        - random_seed (int): Seed for dataset sampling and reproducibility.
        - num_samples (int): Number of dataset examples to process.
        - dataset (str): Dataset identifier passed to load_ds.
        - model_id (str): HuggingFace model ID for the LLM.
        - exp_name (str): Experiment name prefix used in output filenames.
        - task_type (str): Task type passed to construct_prompt.
    """
    experiment_details = {'args': args}
    random.seed(args.random_seed)

    samples = load_ds(args.dataset, seed=args.random_seed, num_samples = args.num_samples)
    
    logging.info('Dataset loaded!')
    
    model_id = args.model_id
    exp_name = args.exp_name
    ds_name = args.dataset
    task_type = args.task_type
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
        
        generated_text, step_sequences, gen_ids, gen_tokens = llm.generate_with_topk(prompt=prompt, temperature = 0.1)
        gen_words, gen_ids_words = generate_words(token_ids=gen_ids, tokenizer=llm.tokenizer)
        
        generations.append({
            'example' : example,
            'topic' : example.get('topic', None),
            'generated_text' : generated_text, 
            'step_sequences' : step_sequences, 
            'gen_ids' : gen_ids, 
            'gen_tokens' : gen_tokens,
            'gen_ids_words' : gen_ids_words,
            'gen_words' : gen_words}
        )
    
    save(generations, f'{exp_name}_{ds_name}_generations.pkl')
    save(experiment_details, f'{exp_name}_{ds_name}_experiment_details.pkl')
    logging.info('Run complete.')
    del llm


if __name__ == '__main__':
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    # First sample generations from LLM.
    logging.info('STARTING `generate_answers`!')
    main(args)
    logging.info('FINISHED `generate_answers`!')