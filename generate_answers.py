"""Sample answers from LLMs on QA task."""
import gc
import logging
import random
import torch

from models.models import * 
from uncertainty_metrics.se import * 
from uncertainty_metrics.pke import *
from utils.subsequences import generate_subsequences
from utils.utils import get_parser, construct_prompt, save, get_metric, setup_logger
from data.utils import load_ds
from compute_uncertainty_measures import main as compute_uq_main


setup_logger()

def main(args):
    experiment_details = {'args': args}
    random.seed(args.random_seed)

    metric = get_metric(args.metric)
    samples = load_ds(args.dataset, seed=args.random_seed, num_samples = args.num_samples)
    
    logging.info('Dataset loaded!')
    
    model_id = args.model_id
    exp_name = args.exp_name
    ds_name = args.dataset
    task_type = args.task_type
    k = args.k
    
    # Initialize model
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
        # print(25 * '-', s, ': ', example)
        
        prompt = construct_prompt(example['question'], task_type=task_type)
        
        generated_text, sampled_tokens, gen_ids = llm.generate_with_topk(prompt=prompt, k = k)
        current_probs, seq_tokens = generate_subsequences(sampled_tokens=sampled_tokens, tokenizer=llm.tokenizer)
        
        acc = metric(generated_text, example, llm)
        
        generations.append({
            'example' : example,
            'acc' : acc, 
            'generated_text' : generated_text, 
            'sampled_tokens' : sampled_tokens, 
            'gen_ids' : gen_ids, 
            'seq_tokens' : seq_tokens, 
            'current_probs' : current_probs}
        )

        # accuracy = np.mean(accuracies)
        # print(f"Overall {dataset_split} split accuracy: {accuracy}")
        # wandb.log({f"{dataset_split}_accuracy": accuracy})
    
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

    # if args.compute_uncertainties:
    #     args.assign_new_wandb_id = False

    # First sample generations from LLM.
    logging.info('STARTING `generate_answers`!')
    main(args)
    logging.info('FINISHED `generate_answers`!')

    if args.compute_uncertainties:
        # Follow with uncertainty calculation script by default.
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(50 * '#X')
        logging.info('STARTING `compute_uncertainty_measures`!')
        compute_uq_main(args)
        logging.info('FINISHED `compute_uncertainty_measures`!')