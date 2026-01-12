"""Sample answers from LLMs on QA task."""
import gc
import logging
import random
import torch

from models.models import * 
from models.nli_models import NLI
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
    
    model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ" #args.model_id
    ellm_model_id = args.ellm_model_id
    exp_name = args.exp_name
    ds_name = args.dataset
    task_type = args.task_type
    k = args.k
    
    # # Initialize model
    llm = LLM(model_id=model_id)
    logging.info('Model init')

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
        
    #     generated_text, sampled_tokens, gen_ids, gen_ids_decoded = llm.generate_with_topk(prompt=prompt, k = k, temperature = 0.9)
    #     current_probs, seq_tokens = generate_subsequences(sampled_tokens=sampled_tokens, tokenizer=llm.tokenizer)
        
    #     generations.append({
    #         'example' : example,
    #         'generated_text' : generated_text, 
    #         'sampled_tokens' : sampled_tokens, 
    #         'gen_ids' : gen_ids, 
    #         'seq_tokens' : seq_tokens, 
    #         'current_probs' : current_probs}
    #     )
        
    # # del llm
    
    # save(generations, f'{exp_name}_{ds_name}_data_generations.pkl')
    # save(experiment_details, f'{exp_name}_{ds_name}_data_details.pkl')
    # logging.info('Run complete.')
    
    generations = load(f'{exp_name}_{ds_name}_data_generations.pkl')
    
    entries = []
    
    # ellm = LLM(model_id=ellm_model_id)
    if "nli" in ellm_model_id: 
        ellm = NLI(model_id=ellm_model_id)
    else:  
        ellm = LLM(model_id=ellm_model_id, storage_type='open_ai_api')

    MAX_BATCH = 32
    for e, element in enumerate(generations): 
        logging.info(element['generated_text'])
        example = element['example']
        question = example['question']
        gen_ids = element['gen_ids']
        seq_tokens = element['seq_tokens']
        sampled_tokens = element['sampled_tokens']
                            
        for s, step in enumerate(seq_tokens): 
            decoded_seqs = step.get('alternative_sequence_decoded', None) 
            decoded_seqs = list(set(decoded_seqs))
            prefix = llm.tokenizer.decode(step['prev_seq'], skip_special_tokens = True)
            
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
                    # else: 
                        #score = ellm.check_implication(string1, string2, question=example, mode = 'data')
                        # score2 = ellm.check_implication(string2, string1, question=example, mode = 'data')
                    
                    # if score1 != score2:
                    #     score = -100000
                    # else: 
                    #     score = score2
                    batched_pairs.append((string1, string2))
                    checked_ids.append((i, j))
                    checked_ids.append((j, i))
                    
            #all_scores = []
            all_labels = []
            for b in range(0, len(batched_pairs), MAX_BATCH):
                sub = batched_pairs[b:b+MAX_BATCH]
                labels = ellm.check_implication_batch(sub, question)
                #all_scores.extend(scores)
                all_labels.extend(labels)
                # all_scores.extend([1000] * len(sub))       
            logging.info(all_labels)    
            # 5. Store results
            for (p1, p2), label in zip(batched_pairs, all_labels):
                entry = {
                    'generated_text': element['generated_text'],
                    'true_answer': element['example']['answer'],
                    'question': question,
                    'index': s,
                    'prefix': prefix,
                    'text1': p1,
                    'text2': p2,
                    #'score': score,
                    'label' : label
                }
                entries.append(entry)
    save(entries, f'{exp_name}_{ds_name}_data_v6.pkl')
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