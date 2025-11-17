"""Sample answers from LLMs on QA task."""
import gc
import logging
import random
import torch

from models.models import * 
from uncertainty_metrics.se import * 
from uncertainty_metrics.pke import *
from utils.subsequences import generate_subsequences
from utils.utils import get_parser, construct_prompt, save, get_metric, setup_logger, load
from data.utils import load_ds
from compute_uncertainty_measures import main as compute_uq_main


setup_logger()

def main(args):
    experiment_details = {'args': args}
    random.seed(args.random_seed)

    samples = load_ds(args.dataset, seed=args.random_seed, num_samples = args.num_samples)
    
    logging.info('Dataset loaded!')
    
    model_id = args.model_id
    ellm_model_id = args.ellm_model_id
    exp_name = args.exp_name
    ds_name = args.dataset
    task_type = args.task_type
    k = args.k
    
    # # Initialize model
    # llm = LLM(model_id=model_id)
    # logging.info('Model init')

    # # Start answer generation.
    # logging.info(80 * '=')
    # logging.info('Generating answers: ')
    # logging.info(80 * '=')
        
    # generations = []

    # it = 0
    # for s, example in enumerate(samples): 
    #     if (it + 1 % 10) == 0:
    #         gc.collect()
    #         torch.cuda.empty_cache()
    #     it += 1
    #     # print(25 * '-', s, ': ', example)
        
    #     prompt = construct_prompt(example['question'], task_type=task_type)
        
    #     generated_text, sampled_tokens, gen_ids = llm.generate_with_topk(prompt=prompt, k = k)
    #     current_probs, seq_tokens = generate_subsequences(sampled_tokens=sampled_tokens, tokenizer=llm.tokenizer)
        
    #     generations.append({
    #         'example' : example,
    #         'generated_text' : generated_text, 
    #         'sampled_tokens' : sampled_tokens, 
    #         'gen_ids' : gen_ids, 
    #         'seq_tokens' : seq_tokens, 
    #         'current_probs' : current_probs}
    #     )
        
    # del llm
    
    # save(generations, f'{exp_name}_{ds_name}_data_generations.pkl')
    # save(experiment_details, f'{exp_name}_{ds_name}__data_details.pkl')
    # logging.info('Run complete.')
    
    generations = load(f'{exp_name}_{ds_name}_data_generations.pkl')
    
    entries = []
    
    ellm = EntailmentLLM(model_id=ellm_model_id)

    
    for e, element in enumerate(generations): 
        logging.info(element['generated_text'])
        example = element['example']
        question = example['question']
        gen_ids = element['gen_ids']
        seq_tokens = element['seq_tokens']
        sampled_tokens = element['sampled_tokens']
                    
        for s, step in enumerate(seq_tokens): 
            decoded_seqs = step.get('s_decoded', None) 
            checked_ids = []               
            for i, string1 in enumerate(decoded_seqs):
                for j in range(i, len(decoded_seqs)):
                    string2 = decoded_seqs[j]
                    if (i, j) in checked_ids: 
                        continue
                    if i == j or string1 == string2: 
                        score = 1
                    else: 
                        score = ellm.check_implication(string1, string2, question=example, mode = 'data')
                        # score2 = ellm.check_implication(string2, string1, question=example, mode = 'data')
                    
                    # if score1 != score2:
                    #     score = -100000
                    # else: 
                    #     score = score2
                    
                    checked_ids.append((i, j))
                    checked_ids.append((j, i))
                    
                    entry = {
                        'generated_text' : element['generated_text'],
                        'true_answer': element['example']['answer'],
                        'question' : question,
                        'index' : s, 
                        'text1' : string1, 
                        'text2' : string2, 
                        'score' : score
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

    # if args.compute_uncertainties:
    #     args.assign_new_wandb_id = False

    # First sample generations from LLM.
    logging.info('STARTING `generate_training_data`!')
    main(args)
    logging.info('FINISHED `generate_training_data`!')