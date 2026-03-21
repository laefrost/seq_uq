import torch
import re
import torch.nn.functional as F
import spacy
import gc
import logging
import random
from sentence_transformers import SentenceTransformer
import nltk
import random

nltk.download('punkt')

from models.models import * 
from models.nli_models import * 
from uncertainty_metrics.se import * 
from uncertainty_metrics.pke import *
from uncertainty_metrics.vne import *
from uncertainty_metrics.rao import rao_entropy, avg_conflict
from utils.subsequences import generate_subsequences, remove_subsequences, generate_word_subsequences
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, paired_cosine_distances, polynomial_kernel, cosine_similarity, laplacian_kernel, rbf_kernel
from utils.utils import get_parser, construct_prompt, save, get_metric, setup_logger, load
from data.utils import load_ds
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import numpy as np

#  --emb_model_id=models_peft/all-MiniLM-L6-v2-peft/final
SEED = 6400
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True) 
setup_logger()

def get_last_fragment(text):
    special_tokens = ['</s>', '<|end|>', '<|imend|>', '<|endoftext|>']  # add more here
    escaped = [re.escape(t) for t in special_tokens]
    pattern = r'[.?!]|' + '|'.join(escaped)
    parts = re.split(pattern, text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts[-1] if parts else text.strip()

def get_truncated_texts(emb_model, texts):
    tokens = emb_model.tokenizer(
        texts,
        truncation=True,
        max_length=emb_model.max_seq_length,
        padding=False,
        return_tensors="pt"
    )
    return [
        emb_model.tokenizer.decode(ids, skip_special_tokens=True)
        for ids in tokens["input_ids"]
    ]


def se_pipe_across_tokens(question, seq_tokens, ellm, mode = 'adapted'): 
    cluster_ids_across_steps, topic_ids_across_steps, probs = generate_semantic_subsequence_ids(seq_tokens=seq_tokens, question = question, ellm=ellm, mode=mode)
    entropies_to = compute_se_across_subsequences(cluster_ids_across_steps=cluster_ids_across_steps, seq_tokens=seq_tokens, probs = probs, mode = 'subsequ')  
    entropies_to_weighted = compute_se_across_subsequences(cluster_ids_across_steps=cluster_ids_across_steps, seq_tokens=seq_tokens, probs = probs, mode = 'subsequ', topics = topic_ids_across_steps)    
  
    return entropies_to, entropies_to_weighted 


def se_pipe_across_words(question, seq_words, ellm, mode = 'adapted'):   
    cluster_ids_across_steps, topic_ids_across_steps, probs = generate_semantic_subsequence_ids(seq_tokens=seq_words, question = question, ellm=ellm, mode=mode)
    entropies_to = compute_se_across_subsequences(cluster_ids_across_steps=cluster_ids_across_steps, seq_tokens=seq_words, probs = probs, mode = 'subsequ')
    entropies_to_weighted = compute_se_across_subsequences(cluster_ids_across_steps=cluster_ids_across_steps, seq_tokens=seq_words, probs = probs, mode = 'subsequ', topics = topic_ids_across_steps)    
    return entropies_to, entropies_to_weighted 


def uq_pipe_across_tokens(seq_tokens, emb_model, question, gen_ids, tokenizer_llm, mode = 'sampling', task_type = 'qa'):
    vnes = []
    vnes_tokens = []
    vnes_add_combined = []
    vnes_multpl_combined = []
    vnes_rbf = []
    stds = []
    for s_index, s in enumerate(seq_tokens): 
        if task_type != 'qa': 
            last_sentences = [nltk.sent_tokenize(text)[-1] for text in s['alternative_sequence_decoded']]
            ems = emb_model.encode([question + ' ' + s for s in last_sentences], normalize_embeddings=True) # 10, 384
        else: 
            ems = emb_model.encode([question + ' ' + s for s in s['alternative_sequence_decoded']], normalize_embeddings=True) # 10, 384
        
        ems_token = emb_model.encode(s['alternative_tokens_str'], normalize_embeddings=True)  
        if max(s['alternative_token_probs']) > 0.99: 
            print("bigger item", max(s['alternative_token_probs']))
            vne_emb_rbf, std_emb = 0, 0
            vne_emb, std_emb =  0, 0
            vne_tokens, std_token =  0, 0
            vne_add_combined, std_add =  0, 0
            vne_multpl_combined, std_joint =  0, 0
            
        else:      
            vne_emb_rbf, std_emb = vne(ems, mode = mode, probs = s['alternative_token_probs'])
            vne_emb, std_emb = vne(ems, kernel=lambda x, y: cosine_similarity(x, y), mode = mode, probs = s['alternative_token_probs']) #, kernel=lambda x, y: cosine_similarity(x, y))
            vne_tokens, std_token = vne(ems_token, kernel=lambda x, y: cosine_similarity(x, y), mode = mode, probs = s['alternative_token_probs']) #, kernel=lambda x, y: cosine_similarity(x, y))
            vne_add_combined, std_add = vne(Y = ems, kernel=lambda x, y: cosine_similarity(x, y), Y2 = ems_token, combination_mode = "additive", mode = mode, probs = s['alternative_token_probs']) #, kernel=lambda x, y: cosine_similarity(x, y))
            vne_multpl_combined, std_joint = vne(ems, kernel=lambda x, y: cosine_similarity(x, y), Y2 = ems_token, combination_mode = "multiplicative", mode = mode, probs = s['alternative_token_probs'])
            
        vnes_tokens.append(vne_tokens)
        vnes.append(vne_emb)
        vnes_add_combined.append(vne_add_combined)
        vnes_multpl_combined.append(vne_multpl_combined)
        vnes_rbf.append(vne_emb_rbf)
        stds.append(std_emb)
        
    return vnes, vnes_tokens, vnes_multpl_combined, vnes_add_combined, vnes_rbf, stds


def uq_pipe_across_words(seq_words, emb_model, mode = 'sampling', question = "", task_type = 'qa'):
    l = 1
    vnes = []
    vnes_combined = []
    vnes_add_combined = []
    vnes_word = []
    vnes_rbf = []
    
    stds = []
    
    for s_index, s in enumerate(seq_words): 
        previous_seq = s['prev_seq_question_decoded']
        alternative_sequences = s['alternative_sequence_question_decoded']
        
        words = []
        for alternative in alternative_sequences:
            word = alternative[len(previous_seq):].lstrip()
            words.append(word)
        if task_type != 'qa':
            last_sentences = [nltk.sent_tokenize(text)[-1] for text in s['alternative_sequence_question_decoded']]
            alternative_sequences_emb = emb_model.encode([question + ' ' + s for s in last_sentences], normalize_embeddings=True)
        else: 
            alternative_sequences_emb = emb_model.encode(alternative_sequences, normalize_embeddings=True)
            
        words_emb = emb_model.encode(words, normalize_embeddings = True)   
        
        if len(alternative_sequences) == 1:
            vne_emb, vne_emb_rbf, vne_word, vne_combined, vne_add, vne_delta, rao_emb, conflict, conflict_ct, std_emb = 0, 0, 0, 0, 0, 0, 0, 0, 0,0
        elif max(s['alternative_token_probs']) > 0.99: 
            vne_emb, vne_emb_rbf, vne_word, vne_combined, vne_add, vne_delta, rao_emb, conflict, conflict_ct, std_emb = 0, 0, 0, 0, 0, 0, 0, 0, 0,0
        else:
            vne_emb_rbf, std_emb_rbf = vne(Y = alternative_sequences_emb, mode = mode, probs = s['alternative_token_probs'])#, kernel=lambda x, y: cosine_similarity(x, y))
            vne_emb, std_emb = vne(Y = alternative_sequences_emb, kernel=lambda x, y: cosine_similarity(x, y), mode = mode, probs = s['alternative_token_probs'])#, kernel=lambda x, y: cosine_similarity(x, y))
            vne_word, std_word = vne(Y = words_emb, kernel=lambda x, y: cosine_similarity(x, y), mode=mode, probs = s['alternative_token_probs']) #, kernel=lambda x, y: cosine_similarity(x, y))#, kernel=lambda x, y: cosine_similarity(x, y), type_mask=mask_pos)
            vne_combined, std_joint = vne(Y = alternative_sequences_emb, kernel=lambda x, y: cosine_similarity(x, y), Y2 = words_emb, combination_mode = "multiplicative", mode = mode, probs = s['alternative_token_probs']) #, kernel=lambda x, y: cosine_similarity(x, y)) #, kernel=lambda x, y: cosine_similarity(x, y), type_mask=mask_pos)
            vne_add, std_add =  vne(Y = alternative_sequences_emb, kernel=lambda x, y: cosine_similarity(x, y), Y2 = words_emb, combination_mode = "additive", mode = mode, probs = s['alternative_token_probs'])
        
        vnes.append(vne_emb)
        vnes_word.append(vne_word)
        vnes_combined.append(vne_combined)
        vnes_add_combined.append(vne_add)
        vnes_rbf.append(vne_emb_rbf)
        stds.append(std_emb)
            
    return vnes, vnes_word, vnes_combined, vnes_add_combined, vnes_rbf, stds         
        
        
def main(args): 
    model_id = args.model_id
    emb_model_id = args.emb_model_id
    emb_model_id_deltas = args.emb_model_id_deltas
    ellm_model_id = args.ellm_model_id
    exp_name = args.exp_name
    ds_name = args.dataset
    consider_types = args.consider_types
    task_type = args.task_type
    
    generations = load(f'{exp_name}_{ds_name}_generations.pkl')
    logging.info('Dataset loaded!')
    logging.info(type(generations))
    uqs_sampled = []    
    uqs_selection = []    

    llm = LLM(model_id=model_id)
    if model_id != ellm_model_id:
        if "nli" in ellm_model_id: 
            ellm = NLI(model_id=ellm_model_id)
        else:  
            ellm = LLM(model_id=ellm_model_id, storage_type='open_ai_api')
    emb_model = SentenceTransformer(emb_model_id) #SentenceTransformer("all-MiniLM-L6-v2")  
    emb_model.tokenizer.truncation_side = "left"  
    emb_model.eval()
    emb_model_deltas = SentenceTransformer(emb_model_id_deltas)
    emb_model_deltas.tokenizer.truncation_side = "left" 
    emb_model_deltas.eval()
    
        
    for e, element in enumerate(generations):
        if task_type == "bio": 
            if 'Phi' in model_id and e in [1, 2]:
                print("Skipped", e)
                continue 
            if 'Qwen' in model_id and e in [1, 2]:
                print("Skipped", e)
                continue 
            if 'Mistral' in model_id and e in [1, 2]:
                print("Skipped", e)
                continue 
        logging.info(element['generated_text'])
        example = element['example']
        question = example['question']
        step_sequences = element['step_sequences']    
        generated_words = element['gen_words']
        generated_tokens = element['gen_tokens']
        gen_ids = element['gen_ids']
        word_ids = element['gen_ids_words']
        generated_text = element['generated_text']
        seq_tokens_sampled = generate_subsequences(step_sequences, llm.tokenizer, gen_ids, sampling_k = 10, scaling_p = None, selection_p = 0.95, method = "sampling", question = question)
        seq_words_sampled = generate_word_subsequences(seq_tokens_sampled, generated_words, word_ids, question, generated_text, llm.tokenizer)
        
        seq_tokens_selection = generate_subsequences(step_sequences, llm.tokenizer, gen_ids, sampling_k = 10, scaling_p = None, selection_p = 0.95, method = "top_k", question = question)
        seq_words_selection = generate_word_subsequences(seq_tokens_selection, generated_words, word_ids, question, generated_text, llm.tokenizer)
                
        if model_id != ellm_model_id: 
            ses_words_to, ses_words_to_w = se_pipe_across_words(example['question'], seq_words_sampled, ellm, mode='adapted')
            ses_tokens_to, ses_tokens_to_w = se_pipe_across_tokens(example['question'], seq_tokens_sampled, ellm, mode='adapted')
        else: 
            ses_words_to, ses_words_to_w = se_pipe_across_words(example['question'], seq_words_sampled, llm)
            ses_tokens_to, ses_tokens_to_w  = se_pipe_across_tokens(example['question'], seq_tokens_sampled, llm)
        
        vnes_token, vnes_token_token, vnes_token_multpl_combined, vnes_token_add_combined, vnes_token_rbf, std_emb_token = uq_pipe_across_tokens(seq_tokens_sampled, emb_model=emb_model, question = example['question'], gen_ids = gen_ids, tokenizer_llm=llm.tokenizer, task_type = task_type)     
        vnes_token_disp, vnes_token_token_disp, vnes_token_multpl_combined_disp, vnes_token_add_combined_disp, vnes_token_disp_rbf, std_emb_token_disp = uq_pipe_across_tokens(seq_tokens_sampled, emb_model=emb_model_deltas, question = example['question'], gen_ids = gen_ids, tokenizer_llm=llm.tokenizer, task_type = task_type)     
        
        vnes_word, vnes_word_word, vnes_word_combined, vnes_word_add, vnes_word_rbf, std_emb_word = uq_pipe_across_words(seq_words_sampled, emb_model=emb_model, question = question, task_type = task_type)
        vnes_word_disp, vnes_word_word_disp, vnes_word_combined_disp, vnes_word_add_disp, vnes_word_disp_rbf, std_emb_word_disp = uq_pipe_across_words(seq_words_sampled, emb_model=emb_model_deltas, question = question, task_type = task_type)
        
        uqs_sampled.append({'question': example['question'], 
                    'gen_text' : element['generated_text'], 
                    'gen_words': generated_words,
                    'gen_tokens' : generated_tokens,
                    'ses_word_to' : ses_words_to,
                    'ses_word_to_w' : ses_words_to_w,
                    'ses_tokens_to' : ses_tokens_to,
                    'ses_tokens_to_w' : ses_tokens_to_w,
                    'vnes_token' : vnes_token, 
                    'vnes_token_rbf' : vnes_token_rbf, 
                    'vnes_token_token' : vnes_token_token, 
                    'vnes_token_add_combined' : vnes_token_add_combined, 
                    'vnes_token_multpl_combined' : vnes_token_multpl_combined,
                    'vnes_word_emb' : vnes_word,
                    'vnes_word_emb_rbf' : vnes_word_rbf,
                    'vnes_word_word' : vnes_word_word,
                    'vnes_word_add_combined' : vnes_word_add, 
                    'vnes_word_multpl_combined' : vnes_word_combined,
                    # --------------------------------- dispersion 
                    'vnes_token_disp' : vnes_token_disp, 
                    'vnes_token_disp_rbf' : vnes_token_disp_rbf, 
                    'vnes_token_token_disp' : vnes_token_token_disp, 
                    'vnes_token_add_combined_disp' : vnes_token_add_combined_disp, 
                    'vnes_token_multpl_combined_disp' : vnes_token_multpl_combined_disp,
                    'vnes_word_emb_disp' : vnes_word_disp,
                    'vnes_word_emb_disp_rbf' : vnes_word_disp_rbf,
                    'vnes_word_word_disp' : vnes_word_word_disp,
                    'vnes_word_add_combined_disp' : vnes_word_add_disp, 
                    'vnes_word_multpl_combined_disp' : vnes_word_combined_disp,
                    # ----------------------------------------------------------
                    'ln_probs_word' : [s['ln_prob'] for s in seq_words_sampled],
                    'ln_probs_token' : [s['ln_prob'] for s in seq_tokens_sampled],
                    'entropies_word' : [s['entropy'] for s in seq_words_sampled],
                    'entropies_token' : [s['entropy'] for s in seq_tokens_sampled],
                    'true_answer' : example['answer'], 
                    'seq_tokens' : seq_tokens_sampled,
                    'seq_words' : seq_words_sampled,
                    'std_emb_word_disp' : std_emb_word_disp,
                    'std_emb_token_disp' : std_emb_token_disp,
                    'std_emb_word' : std_emb_word,
                    'std_emb_token' : std_emb_token
                    })
        # print("...........................", vnes_token, vnes_word)
        # ------------------------------------------------------ selection part
        if model_id != ellm_model_id: 
            ses_words_to, ses_words_to_w = se_pipe_across_words(example['question'], seq_words_selection, ellm, mode='adapted')
            ses_tokens_to, ses_tokens_to_w = se_pipe_across_tokens(example['question'], seq_tokens_selection, ellm, mode='adapted')
        else: 
            ses_words_to, ses_words_to_w = se_pipe_across_words(example['question'], seq_words_selection, llm)
            ses_tokens_to, ses_tokens_to_w  = se_pipe_across_tokens(example['question'], seq_tokens_selection, llm)
        
        vnes_token, vnes_token_token, vnes_token_multpl_combined, vnes_token_add_combined, vnes_token_rbf, std_emb_token = uq_pipe_across_tokens(seq_tokens_selection, emb_model=emb_model, question = example['question'], gen_ids = gen_ids, tokenizer_llm=llm.tokenizer, mode='selection', task_type = task_type)     
        vnes_token_disp, vnes_token_token_disp, vnes_token_multpl_combined_disp, vnes_token_add_combined_disp, vnes_token_disp_rbf, std_emb_token_disp = uq_pipe_across_tokens(seq_tokens_selection, emb_model=emb_model_deltas, question = example['question'], gen_ids = gen_ids, tokenizer_llm=llm.tokenizer, mode='selection', task_type = task_type)     

        vnes_word, vnes_word_word, vnes_word_combined, vnes_word_add, vnes_word_rbf, std_emb_word = uq_pipe_across_words(seq_words_selection, emb_model=emb_model, mode='selection', question = question, task_type = task_type)
        vnes_word_disp, vnes_word_word_disp, vnes_word_combined_disp, vnes_word_add_disp, vnes_word_disp_rbf, std_emb_word_disp = uq_pipe_across_words(seq_words_selection, emb_model=emb_model_deltas, mode='selection', question = question, task_type = task_type)

        uqs_selection.append({'question': example['question'], 
                    'gen_text' : element['generated_text'], 
                    'gen_words': generated_words,
                    'gen_tokens' : generated_tokens,
                    'ses_word_to' : ses_words_to,
                    'ses_word_to_w' : ses_words_to_w,
                    'ses_tokens_to' : ses_tokens_to,
                    'ses_tokens_to_w' : ses_tokens_to_w,
                    'vnes_token' : vnes_token, 
                    'vnes_token_rbf' : vnes_token_rbf, 
                    'vnes_token_token' : vnes_token_token, 
                    'vnes_token_add_combined' : vnes_token_add_combined, 
                    'vnes_token_multpl_combined' : vnes_token_multpl_combined,
                    'vnes_word_emb' : vnes_word,
                    'vnes_word_emb_rbf' : vnes_word_rbf,
                    'vnes_word_word' : vnes_word_word,
                    'vnes_word_add_combined' : vnes_word_add, 
                    'vnes_word_multpl_combined' : vnes_word_combined,
                    # --------------------------------- dispersion 
                    'vnes_token_disp' : vnes_token_disp, 
                    'vnes_token_disp_rbf' : vnes_token_disp_rbf, 
                    'vnes_token_token_disp' : vnes_token_token_disp, 
                    'vnes_token_add_combined_disp' : vnes_token_add_combined_disp, 
                    'vnes_token_multpl_combined_disp' : vnes_token_multpl_combined_disp,
                    'vnes_word_emb_disp' : vnes_word_disp,
                    'vnes_word_emb_disp_rbf' : vnes_word_disp_rbf,
                    'vnes_word_word_disp' : vnes_word_word_disp,
                    'vnes_word_add_combined_disp' : vnes_word_add_disp, 
                    'vnes_word_multpl_combined_disp' : vnes_word_combined_disp,
                    # ----------------------------------------------------------
                    'ln_probs_word' : [s['ln_prob'] for s in seq_words_selection],
                    'ln_probs_token' : [s['ln_prob'] for s in seq_tokens_selection],
                    'entropies_word' : [s['entropy'] for s in seq_words_selection],
                    'entropies_token' : [s['entropy'] for s in seq_tokens_selection],
                    'true_answer' : example['answer'], 
                    'seq_tokens' : seq_tokens_selection,
                    'seq_words' : seq_words_selection,
                    'std_emb_word_disp' : std_emb_word_disp,
                    'std_emb_token_disp' : std_emb_token_disp,
                    'std_emb_word' : std_emb_word,
                    'std_emb_token' : std_emb_token
                    })
        print("...........................", vnes_token, vnes_word)
    
    if model_id != ellm_model_id:
        del ellm
    del llm
    save(uqs_sampled, f'{exp_name}_{ds_name}_uqs_sampled.pkl')
    save(uqs_selection, f'{exp_name}_{ds_name}_uqs_selection.pkl')

    
    
        
if __name__ == '__main__':
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    logging.info('STARTING `compute_uncertainty`!')
    main(args)
    logging.info('FINISHED `compute_uncertainty`!')