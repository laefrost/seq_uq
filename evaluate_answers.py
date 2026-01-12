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


setup_logger()

def main(args):
    experiment_details = {'args': args}
    eval_model_id = args.eval_model_id
    exp_name = args.exp_name
    ds_name = args.dataset
    eval_type = args.eval_type
    task_type = args.task_type
    fact_model_name = args.fact_model_name

    
    generations = load(f'{exp_name}_{ds_name}_generations.pkl')
    logging.info('Answers loaded!')
    
    metric = get_metric(args.metric)

    llm_eval = LLM(model_id=eval_model_id, storage_type='hf_inference')
    
    eval_results = []
    
    if eval_type == 'fact_score': 
        fs = FactScorer(model_name= fact_model_name)
        
        generated_answers = []
        topics = []
        true_answers = []
        questions = []
        for gen in generations: 
            generated_answers.append(gen['generated_text'])
            topics.append(gen['topic'])
            questions.append(gen['example']['question'])
            true_answers.append(gen['example']['answer']['aliases'])
        
        result = fs.get_score(topics=topics,
                       generations=generated_answers,
                       true_answers = None, 
                       questions = None, 
                       atomic_facts=None,
                       knowledge_source=None,
                       verbose=True)
        print(result['scores'])
        print('------------------------------')
        print(result['decisions'])
        print('------------------------------')
        print(result['sentences'])
        print('------------------------------')
        assert len(result['decisions']) == len(generations)
        for decision, gen in zip(result['decisions'], generations):
            example = gen['example']
            generated_text = gen['generated_text']
            gen_tokens = gen['gen_tokens']
            d_list = []
            print(decision)
            for d in decision: 
                pattern = r"\(|\)|[0-9]+(?:[.,-][0-9]+)*|[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)*|[.,;?!:]|\n|</s>|'|\"|`|´|-"
                gen_words = re.findall(pattern, d['sentence'])
                if d['is_supported']: 
                    acc_words = ["no"] * len(gen_words)
                    #print(['!'] * 100)
                    #acc_words = llm_eval.check_positions(example['question'], generated_text, example['answer']['aliases'], gen_words)
                else:
                    acc_words = llm_eval.check_positions(example['question'], generated_text, example['answer']['aliases'], gen_words)
                d_list.append({'acc_words' : acc_words, 
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
            
    else: 
        for gen in generations: 
            generated_text = gen['generated_text']
            example = gen['example']
            gen_tokens = gen['gen_tokens']
            gen_words = gen['gen_words']
            if task_type == 'qa':
                acc = metric(generated_text, example, llm_eval)
                print(example['question'], generated_text, example['answer'])    
                if acc == 0:
                    logging.info('wrong answer: ')
                    # acc_tokens = llm_eval.check_positions(example['question'], generated_text, example['answer']['aliases'], gen_tokens)
                    acc_words = llm_eval.check_positions(example['question'], generated_text, example['answer']['aliases'], gen_words)
                else: 
                    #acc_tokens = ["no"] * len(gen_tokens)
                    acc_words = ["no"] * len(gen_words)
            else:
                acc_words = None
            supported = acc != 0
            eval_results.append({
                    'question': example['question'], 
                    'true_answer' : example['answer'], 
                    'gen_text' : generated_text, 
                    'gen_tokens' : gen_tokens, 
                    'gen_words' : gen_words,
                    'acc_facts' : [{'acc_words' : acc_words, 'supported' : supported, 'sentence' : generated_text, 'fact' : generated_text, 'gen_words' : gen_words}], 
                    })
    
    save(eval_results, f'{exp_name}_{ds_name}_evals.pkl')
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