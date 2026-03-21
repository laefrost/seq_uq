"""Sample answers from LLMs on QA task."""
import gc
import logging
import random
import torch

from models.models import * 
from uncertainty_metrics.se import * 
from uncertainty_metrics.pke import *
from utils.subsequences import generate_words
from utils.utils import get_parser, construct_prompt, save, get_metric, setup_logger
from data.utils import load_ds
#from compute_uncertainty_measures import main as compute_uq_main
#from evaluate_answers import main as eval_answers_main
from utils.utils import load




setup_logger()

def main(args):
    experiment_details = {'args': args}
    random.seed(args.random_seed)

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
    # generations_old = load(f'{exp_name}_{ds_name}_generations.pkl')

    it = 0
    for s, example in enumerate(samples): 
    #for s, example in enumerate(generations_old):
        if (it + 1 % 10) == 0:
            gc.collect()
            torch.cuda.empty_cache()
        it += 1        
        prompt = construct_prompt(example['question'], task_type=task_type)
        
        generated_text, step_sequences, gen_ids, gen_tokens = llm.generate_with_topk(prompt=prompt, k = k, temperature = 0.1)
        gen_words, gen_ids_words = generate_words(token_ids=gen_ids, tokenizer=llm.tokenizer)
        
        # generated_text, step_sequences, gen_ids, gen_tokens = example['generated_text'], example['step_sequences'], example['gen_ids'], example['gen_tokens'], 
        # gen_words, gen_ids_words = generate_words(token_ids=gen_ids, tokenizer=llm.tokenizer)
        generations.append({
            'example' : example,
            #'example' : example['example'],
            'topic' : example.get('topic', None),
            #'topic' : example['example'].get('topic', None),
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

    # if args.compute_uncertainties:
    #     args.assign_new_wandb_id = False

    # First sample generations from LLM.
    logging.info('STARTING `generate_answers`!')
    main(args)
    logging.info('FINISHED `generate_answers`!')
    
    if args.eval_answers:
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(50 * '#X')
        logging.info('STARTING `eval_answers`!')
        eval_answers_main(args)
        logging.info('FINISHED `eval_answers`!')

    if args.compute_uncertainties:
        # Follow with uncertainty calculation script by default.
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(50 * '#X')
        logging.info('STARTING `compute_uncertainty_measures`!')
        compute_uq_main(args)
        logging.info('FINISHED `compute_uncertainty_measures`!')