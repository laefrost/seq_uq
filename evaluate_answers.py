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
from factscore.factscorer import FactScorer
import spacy
import re


setup_logger()


def get_depth(token): 
    d = 0
    while token.head != token: 
        token = token.head
        d += 1
    return d

def main(args):
    experiment_details = {'args': args}
    eval_model_id = args.eval_model_id
    exp_name = args.exp_name
    ds_name = args.dataset
    eval_type = args.eval_type
    task_type = args.task_type
    fact_model_name = args.fact_model_name

    nlp = spacy.load("en_core_web_sm")
    generations = load(f'results/{exp_name}_{ds_name}_generations.pkl')
    uqs = load(f'results/{exp_name}_{ds_name}_uqs_all-MiniLM-L6-v2.pkl.pkl')
    logging.info('Answers loaded!')
    
    metric = get_metric(args.metric)

    if "gpt" in eval_model_id: 
        llm_eval = LLM(model_id=eval_model_id, storage_type='open_ai_api')
    else: 
        llm_eval = LLM(model_id=eval_model_id, storage_type='hf_inference')
    
    eval_results = []
    
    if eval_type == 'fact_score':
    # if task_type != 'qa': 
        fs = FactScorer(model_name= fact_model_name)
        
        generated_answers = []
        topics = []
        true_answers = []
        questions = []
        atomic_facts = []
        roles_words = []
        subtrees_words = []
        depths_words = []
        sem_rels_words = []
        for gen, u in enumerate(generations, uqs):
            example = gen['example']
            gen_words = u['gen_words']
            generated_answers.append(gen['generated_text'])
            topics.append(gen['topic'])
            questions.append(gen['example']['question'])
            true_answers.append(gen['example']['answer']['aliases'])
            atomic_facts.append([gen['generated_text']])

            doc = nlp(gen['generated_text'])
            role_words = [token.dep_ for token in doc]
            roles_words.append(role_words)
            
            subtree_words = [len(list(token.subtree)) for token in doc]
            subtrees_words.append(subtree_words)
            
            depth_words = [get_depth(token) for token in doc]
            depths_words.append(depth_words)
            
            try: 
                sem_rel_words = llm_eval.check_positions(example['question'], gen['generated_text'], example['answer']['aliases'], gen_words, mode='get_se_imp')
            except Exception as e:
                print("error in sem_rel_words", e)
                sem_rel_words = [None]
            # sem_rel_words =['tmp_value']
            # sem_rels_words.append(sem_rel_words)
        
        if task_type != 'qa': atomic_facts = None
        result = fs.get_score(topics=topics,
                       generations=generated_answers,
                       true_answers = None, 
                       questions = None, 
                       atomic_facts=None,
                       knowledge_source=None,
                       verbose=True)
        
        
        assert len(result['decisions']) == len(generations)
        
        for decision, gen, role_words, subtree_words, depth_words, sem_rel_words  in zip(result['decisions'], generations, roles_words, subtrees_words, depths_words, sem_rels_words):
            example = gen['example']
            generated_text = gen['generated_text']
            gen_tokens = gen['gen_tokens']
            d_list = []
            print(decision)
            for d in decision: 
                
                
                # TODO: gen_words this needs to be different (probably based on tokenizer as in subsequences)
                
                #pattern = r"\(|\)|[0-9]+(?:[.,-][0-9]+)*|[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)*|[.,;?!:]|\n|</s>|'|\"|`|´|-"
                # gen_words = re.findall(pattern, d['sentence'])
                # doc = nlp(d['sentence'])
                # role_words = [token.dep_ for token in doc]
                # subtrees_words = [len(list(token.subtree)) for token in doc]
                # depth_words = [get_depth(token) for token in doc]
                # sem_rel_words = llm_eval.check_positions(example['question'], generated_text, example['answer']['aliases'], gen_words, mode='get_se_imp')
                if d['is_supported']: 
                    acc_words = ["no"] * len(gen_words)
                else:
                    try:
                        print("checking positins")
                        # Evlt muss man das gar nicht anpassen dann mapped das halt auf die ganze Sequenz 
                        acc_words = llm_eval.check_positions(example['question'], generated_text, example['answer']['aliases'], gen_words)
                    except Exception as e:
                        print("Error acc_words: --------------",e) 
                        acc_words = None
                d_list.append({'acc_words' : acc_words, 
                               'sem_rel_words' : sem_rel_words,
                               'role_words' : role_words,
                               'subtree_words' : subtrees_words, 
                               'depth_words' : depth_words,
                               'supported' : d['is_supported'], 
                               'sentence' : d['sentence'], 
                               'fact' : d['atom'], 
                               'matched_words' : d['matched_words'], 
                               'matched indices' : d['matched_word_indices'],
                               'gen_words' : gen_words})
                
            eval_results.append({
                    'question': example['question'], 
                    'true_answer' : example['answer'], 
                    'gen_text' : generated_text, 
                    'gen_tokens' : gen_tokens, 
                    'gen_words' : gen_words,
                    'acc_facts' : d_list, 
                    })
    # else: 
    #     # TODO: adapt this to structure from above, 
    #     for gen in generations: 
    #         generated_text = gen['generated_text']
    #         example = gen['example']
    #         gen_tokens = gen['gen_tokens']
    #         gen_words = gen['gen_words']
    #         if task_type == 'qa':
    #             acc = metric(generated_text, example, llm_eval)
    #             print(example['question'], generated_text, example['answer'])    
    #             if acc == 0:
    #                 logging.info('wrong answer: ')
    #                 # acc_tokens = llm_eval.check_positions(example['question'], generated_text, example['answer']['aliases'], gen_tokens)
    #                 acc_words = llm_eval.check_positions(example['question'], generated_text, example['answer']['aliases'], gen_words)
    #             else: 
    #                 #acc_tokens = ["no"] * len(gen_tokens)
    #                 acc_words = ["no"] * len(gen_words)
    #         else:
    #             acc_words = None
    #         sem_rel_words = llm_eval.check_positions(example['question'], generated_text, example['answer']['aliases'], gen_words, mode='get_se_imp')
    #         supported = acc != 0
    #         doc = nlp(generated_text)
    #         role_words = [token.dep_ for token in doc]
    #         subtrees_words = [len(list(token.subtree)) for token in doc]
    #         depth_words = [get_depth(token) for token in doc]
    #         acc_facts = {'acc_words' : acc_words, 
    #                            'role_words' : role_words,
    #                            'subtree_words' : subtrees_words, 
    #                            'depth_words' : depth_words,
    #                            'supported' : supported, 
    #                            'sentence' : generated_text, 
    #                            'fact' : generated_text, 
    #                            'matched_words' : d['matched_words'], 
    #                            'matched indices' : d['matched_word_indices'],
    #                            'gen_words' : gen_words}
            
    #         eval_results.append({
    #                 'question': example['question'], 
    #                 'true_answer' : example['answer'], 
    #                 'gen_text' : generated_text, 
    #                 'gen_tokens' : gen_tokens, 
    #                 'gen_words' : gen_words,
    #                 'acc_facts' : acc_facts
    #                 })
    
    save(eval_results, f'{exp_name}_{ds_name}_evals_atomic_facts.pkl')
    logging.info('Run complete.')
    del llm_eval


if __name__ == '__main__':
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Evaluating answers with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    logging.info('STARTING `evaluate_answers`!')
    main(args)
    logging.info('FINISHED `evaluate_answers`!')