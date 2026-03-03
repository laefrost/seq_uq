import torch
import re
import torch.nn.functional as F
import spacy
import gc
import logging
import random
from sentence_transformers import SentenceTransformer

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


setup_logger()

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


def uq_pipe_across_tokens(seq_tokens, emb_model, question, gen_ids, tokenizer_llm, mode = 'sampling'):
    print("Mode: ", mode)
    vnes = []
    vnes_tokens = []
    vnes_add_combined = []
    vnes_multpl_combined = []
    vnes_deltas = []
    ln_probs = []
    raos = []
    conflicts = []
    stds = []
    for s_index, s in enumerate(seq_tokens): 
        ln_prob = - np.log(s['current_prob'])
        ln_probs.append(ln_prob)
        # Attach Question to generated prefix
        ems = emb_model.encode([question + ' ' + s for s in s['alternative_sequence_decoded']], normalize_embeddings=True) # 10, 384
        #ems_deltas = emb_model_deltas.encode([question + ' ' + s for s in s['alternative_sequence_decoded']], normalize_embeddings=True)
        ems_token = emb_model.encode(s['alternative_tokens_str'], normalize_embeddings=True)
        prev_seq = question + tokenizer_llm.decode(gen_ids[:s_index], skip_special_tokens=True)
        prev_ems = emb_model.encode(prev_seq, normalize_embeddings=True)
        #inputs = tokenizer_emb([question + ' ' + s for s in s['alternative_sequence_decoded']], return_tensors="pt", padding=True).to('cuda')
        #outputs = emb_model(input = inputs)#, output_hidden_states=True)
        #token_embs = outputs.token_embeddings
        
        #last_token_indices = outputs.attention_mask.sum(dim=1) - 1  # [batch]
        #last_valid_embs = token_embs[torch.arange(len(last_token_indices)),  # batch dimension (0..B-1)
        #                             last_token_indices-1                      # different token index per batch
        #                             ].detach().cpu().numpy()
        
        embedding_deltas = prev_ems - ems
        #embedding_deltas = embedding_deltas / (np.linalg.norm(embedding_deltas, axis=1, keepdims=True) + 1e-12)
        
        #ke_deltas = kernel_noise(embedding_deltas, kernel=lambda x, y: cosine_similarity(x, y))
        
        
        #vne_deltas = vne(embedding_deltas) #, kernel=lambda x, y: cosine_similarity(x, y))
        
        
        vne_emb, std_emb = vne(ems, mode = mode, probs = s['alternative_token_probs']) #, kernel=lambda x, y: cosine_similarity(x, y))
        vne_tokens, std_token = vne(ems_token, mode = mode, probs = s['alternative_token_probs']) #, kernel=lambda x, y: cosine_similarity(x, y))
        vne_add_combined, std_add = vne(Y = ems, Y2 = ems_token, combination_mode = "additive", mode = mode, probs = s['alternative_token_probs']) #, kernel=lambda x, y: cosine_similarity(x, y))
        vne_multpl_combined, std_joint = vne(ems, Y2 = ems_token, combination_mode = "multiplicative", mode = mode, probs = s['alternative_token_probs'])
        vne_deltas, std_deltas = vne(embedding_deltas, mode = mode, probs = s['alternative_token_probs'])
        rao_emb = rao_entropy(Y = ems, probs = s['alternative_token_probs'])
        conflict = avg_conflict(Y = ems, probs = s['alternative_token_probs'])
        
        # vne_tokens, std_token = vne(ems, mode = mode, probs = s['alternative_token_probs'], kernel=lambda x, y: rbf_kernel(x, y, gamma=1/10))
        # vne_add_combined, std_add = vne(ems, mode = mode, probs = s['alternative_token_probs'], kernel=lambda x, y: rbf_kernel(x, y, gamma=1/384))
        # vne_multpl_combined, std_joint = vne(ems, mode = mode, probs = s['alternative_token_probs'], kernel=lambda x, y: rbf_kernel(x, y, gamma=1/1000))
        # vne_deltas, std_deltas = vne(ems, mode = mode, probs = s['alternative_token_probs'], kernel=lambda x, y: rbf_kernel(x, y, gamma=2.5))
        # rao_emb = vne(ems, mode = mode, probs = s['alternative_token_probs'], kernel=lambda x, y: rbf_kernel(x, y, gamma=5))
        # conflict = vne(ems, mode = mode, probs = s['alternative_token_probs'], kernel=lambda x, y: rbf_kernel(x, y, gamma=0))
        
        vnes_tokens.append(vne_tokens)
        vnes.append(vne_emb)
        vnes_add_combined.append(vne_add_combined)
        vnes_multpl_combined.append(vne_multpl_combined)
        vnes_deltas.append(vne_deltas)
        raos.append(rao_emb)
        conflicts.append(conflict)
        stds.append(std_emb)
        
    return vnes, vnes_tokens, vnes_add_combined, vnes_multpl_combined, vnes_deltas, ln_probs, raos, conflicts, stds


def uq_pipe_across_words(seq_words, emb_model, mode = 'sampling'):
    print("Mode: ", mode)
    l = 1
    vnes = []
    vnes_ct = []
    vnes_combined = []
    vnes_word = []
    vnes_proj = []
    ln_probs = []
    raos = []
    conflicts = []
    conflicts_ct = []
    vnes_deltas = []
    stds = []
    
    nlp = spacy.load("en_core_web_sm")
    for s_index, s in enumerate(seq_words): 
        previous_seq = s['prev_seq_question_decoded']
        alternative_sequences = s['alternative_sequence_question_decoded']
        
        # print(alternative_sequences)
        
        ln_prob = - np.log(s['current_prob'])
        ln_probs.append(ln_prob)
        
        words = []
        for alternative in alternative_sequences:
            word = alternative[len(previous_seq):].lstrip()
            words.append(word)
        
        emb_model.eval()
        #emb_model_deltas.eval()
        #with torch.no_grad():
        previous_emb = emb_model.encode(previous_seq, normalize_embeddings = True)
        # device = next(emb_model[0].auto_model.parameters()).device
        # inputs = emb_model.tokenizer(alternative_sequences, return_tensors="pt", padding=True).to(device)            
        # # current_seq_tokens = emb_model_embs.tokenizer(current_sequence, return_tensors="pt", padding=True).to(device)
        # alternative_sequences_emb = emb_model(inputs)['sentence_embedding']
        # alternative_sequences_deltas = emb_model_deltas(inputs)['sentence_embedding']
        alternative_sequences_emb = emb_model.encode(alternative_sequences, normalize_embeddings = True)
        words_emb = emb_model.encode(words, normalize_embeddings = True)       
        delta_embs = previous_emb - alternative_sequences_emb
        # delta_embs = delta_embs / (np.linalg.norm(delta_embs, axis=1, keepdims=True) + 1e-12)        
        
        #if consider_types: 
            # go through sequences, do pos_tagging, select last word
        word_pos = []
        for seq in alternative_sequences: 
            doc = nlp(seq)
            word_pos.append(doc[-1].pos_)
        word_pos = np.array(word_pos)
        # mask_pos = (word_pos[:, None] != word_pos).astype(int)
        special_mask = np.isin(word_pos, ["PUNCT", "SPACE"])
        # mask[i, j] = 1 if either i or j is PUNCT/SPACE, else 0
        mask_pos = (special_mask[:, None] | special_mask[None, :]).astype(int)
        
        if len(alternative_sequences) == 1:
            vne_emb, vne_ct, vne_word, vne_combined, vne_add, vne_delta, rao_emb, conflict, conflict_ct, std_emb = 0, 0, 0, 0, 0, 0, 0, 0, 0,0
        else:
            vne_emb, std_emb = vne(Y = alternative_sequences_emb, mode = mode, probs = s['alternative_token_probs'])#, kernel=lambda x, y: cosine_similarity(x, y))
            vne_ct, std_ct = vne(Y = alternative_sequences_emb, type_mask=mask_pos, mode = mode, probs = s['alternative_token_probs'])#, kernel=lambda x, y: cosine_similarity(x, y))
            vne_word, std_word = vne(Y = words_emb, mode=mode, probs = s['alternative_token_probs']) #, kernel=lambda x, y: cosine_similarity(x, y))#, kernel=lambda x, y: cosine_similarity(x, y), type_mask=mask_pos)
            vne_combined, std_joint = vne(Y = alternative_sequences_emb, Y2 = words_emb, combination_mode = "multiplicative", mode = mode, probs = s['alternative_token_probs']) #, kernel=lambda x, y: cosine_similarity(x, y)) #, kernel=lambda x, y: cosine_similarity(x, y), type_mask=mask_pos)
            # vne_proj = vne(Y = alternative_sequences_emb, word_pos = word_pos, kernel=lambda x, y: cosine_similarity(x, y))
            vne_add, std_add =  vne(Y = alternative_sequences_emb, Y2 = words_emb, combination_mode = "additive", mode = mode, probs = s['alternative_token_probs'])
            rao_emb = rao_entropy(Y = alternative_sequences_emb, probs = s['alternative_token_probs']) 
            conflict = avg_conflict(Y = alternative_sequences_emb, probs = s['alternative_token_probs']) 
            conflict_ct = avg_conflict(Y = alternative_sequences_emb, probs = s['alternative_token_probs'], type_mask = mask_pos) 
            vne_delta, std_delta = vne(Y = delta_embs, kernel=lambda x, y: cosine_similarity(x, y), mode = mode, probs = s['alternative_token_probs'])
            
            # vne_ct, std_ct = vne(Y = alternative_sequences_emb, mode = mode, probs = s['alternative_token_probs'], kernel = lambda x, y: rbf_kernel(x, y, gamma=1/10))#, kernel=lambda x, y: cosine_similarity(x, y))
            # vne_word, std_word = vne(Y = alternative_sequences_emb, mode = mode, probs = s['alternative_token_probs'], kernel = lambda x, y: rbf_kernel(x, y, gamma=1/384))
            # vne_combined, std_joint = vne(Y = alternative_sequences_emb, mode = mode, probs = s['alternative_token_probs'], kernel = lambda x, y: rbf_kernel(x, y, gamma=5))
            # vne_add, std_add =  vne(Y = alternative_sequences_emb, mode = mode, probs = s['alternative_token_probs'], kernel = lambda x, y: rbf_kernel(x, y, gamma=10))
            # conflict = vne(Y = alternative_sequences_emb, mode = mode, probs = s['alternative_token_probs'], kernel = lambda x, y: rbf_kernel(x, y, gamma=1/1000))
            # conflict_ct = vne(Y = alternative_sequences_emb, mode = mode, probs = s['alternative_token_probs'], kernel = lambda x, y: rbf_kernel(x, y, gamma=0))
            # vne_delta, std_delta = vne(Y = alternative_sequences_emb, mode = mode, probs = s['alternative_token_probs'], kernel = lambda x, y: rbf_kernel(x, y, gamma=50))
            # rao_emb = vne(Y = alternative_sequences_emb, mode = mode, probs = s['alternative_token_probs'], kernel = lambda x, y: rbf_kernel(x, y, gamma=1/100))

            
        vnes.append(vne_emb)
        vnes_ct.append(vne_ct)
        vnes_word.append(vne_word)
        vnes_combined.append(vne_combined)
        vnes_proj.append(vne_add)
        raos.append(rao_emb)
        conflicts.append(conflict)
        conflicts_ct.append(conflict_ct)
        vnes_deltas.append(vne_delta)
        stds.append(std_emb)
            
    return vnes, vnes_ct, vnes_word, vnes_combined, vnes_proj, vnes_deltas, ln_probs, raos, conflicts, conflicts_ct, stds         
        
        
def main(args): 
    model_id = args.model_id
    emb_model_id = args.emb_model_id
    emb_model_id_deltas = args.emb_model_id_deltas
    ellm_model_id = args.ellm_model_id
    exp_name = args.exp_name
    ds_name = args.dataset
    consider_types = args.consider_types
    
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
    
    emb_model_deltas = SentenceTransformer(emb_model_id_deltas)
    emb_model_deltas.tokenizer.truncation_side = "left" 
        
    for e, element in enumerate(generations):
        logging.info(element['generated_text'])
        example = element['example']
        question = example['question']
        step_sequences = element['step_sequences']    
        generated_words = element['gen_words']
        generated_tokens = element['gen_tokens']
        gen_ids = element['gen_ids']
        word_ids = element['gen_ids_words']
        generated_text = element['generated_text']
        
        seq_tokens_sampled = generate_subsequences(step_sequences, llm.tokenizer, gen_ids, sampling_k = 10, scaling_p = None, selection_p = 0.98, method = "sampling", question = question)
        seq_words_sampled = generate_word_subsequences(seq_tokens_sampled, generated_words, word_ids, question, generated_text, llm.tokenizer)
        
        seq_tokens_selection = generate_subsequences(step_sequences, llm.tokenizer, gen_ids, sampling_k = 10, scaling_p = None, selection_p = 0.98, method = "selection", question = question)
        seq_words_selection = generate_word_subsequences(seq_tokens_selection, generated_words, word_ids, question, generated_text, llm.tokenizer)
                
        if model_id != ellm_model_id: 
            ses_words_to, ses_words_to_w = se_pipe_across_words(example['question'], seq_words_sampled, ellm, mode='adapted')
            ses_tokens_to, ses_tokens_to_w = se_pipe_across_tokens(example['question'], seq_tokens_sampled, ellm, mode='adapted')
        else: 
            ses_words_to, ses_words_to_w = se_pipe_across_words(example['question'], seq_words_sampled, llm)
            ses_tokens_to, ses_tokens_to_w  = se_pipe_across_tokens(example['question'], seq_tokens_sampled, llm)
        
        vnes_token, vnes_token_token, vnes_token_add_combined, vnes_token_multpl_combined, vnes_token_deltas, ln_probs_token, raos_token, conflicts_token, std_emb_token = uq_pipe_across_tokens(seq_tokens_sampled, emb_model=emb_model, question = example['question'], gen_ids = gen_ids, tokenizer_llm=llm.tokenizer)     
        vnes_token_disp, vnes_token_token_disp, vnes_token_add_combined_disp, vnes_token_multpl_combined_disp, vnes_token_deltas_disp, ln_probs_token_disp, raos_token_disp, conflicts_token_disp, std_emb_token_disp = uq_pipe_across_tokens(seq_tokens_sampled, emb_model=emb_model_deltas, question = example['question'], gen_ids = gen_ids, tokenizer_llm=llm.tokenizer)     
        
        vnes_word, vnes_word_ct, vnes_word_word, vnes_word_combined, vnes_word_proj, vnes_word_deltas, ln_probs_word, raos_word, conflicts_word, conflicts_word_ct, std_emb_word = uq_pipe_across_words(seq_words_sampled, emb_model=emb_model)
        vnes_word_disp, vnes_word_ct_disp, vnes_word_word_disp, vnes_word_combined_disp, vnes_word_proj_disp, vnes_word_deltas_disp, ln_probs_word_disp, raos_word_disp, conflicts_word_disp, conflicts_word_ct_disp, std_emb_word_disp = uq_pipe_across_words(seq_words_sampled, emb_model=emb_model_deltas)
        
        
        uqs_sampled.append({'question': example['question'], 
                    'gen_text' : element['generated_text'], 
                    'gen_words': generated_words,
                    'gen_tokens' : generated_tokens,
                    'ses_word_to' : ses_words_to,
                    'ses_word_to_w' : ses_words_to_w,
                    'ses_tokens_to' : ses_tokens_to,
                    'ses_tokens_to_w' : ses_tokens_to_w,
                    'vnes_token' : vnes_token, 
                    'vnes_token_token' : vnes_token_token, 
                    'vnes_token_add_combined' : vnes_token_add_combined, 
                    'vnes_token_multpl_combined' : vnes_token_multpl_combined,
                    'vnes_token_deltas': vnes_token_deltas,
                    'vnes_word_emb' : vnes_word,
                    'vnes_word_emb_ct' : vnes_word_ct, 
                    'vnes_word_word' : vnes_word_word,
                    'vnes_word_add_combined' : vnes_word_proj, 
                    'vnes_word_multpl_combined' : vnes_word_combined,
                    'vnes_word_deltas' : vnes_word_deltas, 
                    'raos_token' : raos_token, 
                    'raos_word' : raos_word,
                    'conflicts_word': conflicts_word, 
                    'conflicts_word_ct' : conflicts_word_ct, 
                    'conflicts_token' : conflicts_token,
                    # --------------------------------- dispersion 
                    'vnes_token_disp' : vnes_token_disp, 
                    'vnes_token_token_disp' : vnes_token_token_disp, 
                    'vnes_token_add_combined_disp' : vnes_token_add_combined_disp, 
                    'vnes_token_multpl_combined_disp' : vnes_token_multpl_combined_disp,
                    'vnes_token_deltas_disp': vnes_token_deltas_disp,
                    'vnes_word_emb_disp' : vnes_word_disp,
                    'vnes_word_emb_ct_disp' : vnes_word_ct_disp, 
                    'vnes_word_word_disp' : vnes_word_word_disp,
                    'vnes_word_add_combined_disp' : vnes_word_proj_disp, 
                    'vnes_word_multpl_combined_disp' : vnes_word_combined_disp,
                    'vnes_word_deltas_disp' : vnes_word_deltas_disp, 
                    'raos_token_disp' : raos_token_disp, 
                    'raos_word_disp' : raos_word_disp,
                    'conflicts_word_disp': conflicts_word_disp, 
                    'conflicts_word_ct_disp' : conflicts_word_ct_disp, 
                    'conflicts_token_disp' : conflicts_token_disp,
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
        # print("...........................", vnes_token, vnes_word)
        # ------------------------------------------------------ selection part
        if model_id != ellm_model_id: 
            ses_words_to, ses_words_to_w = se_pipe_across_words(example['question'], seq_words_selection, ellm, mode='adapted')
            ses_tokens_to, ses_tokens_to_w = se_pipe_across_tokens(example['question'], seq_tokens_selection, ellm, mode='adapted')
        else: 
            ses_words_to, ses_words_to_w = se_pipe_across_words(example['question'], seq_words_selection, llm)
            ses_tokens_to, ses_tokens_to_w  = se_pipe_across_tokens(example['question'], seq_tokens_selection, llm)
        
        vnes_token, vnes_token_token, vnes_token_add_combined, vnes_token_multpl_combined, vnes_token_deltas, ln_probs_token, raos_token, conflicts_token, std_emb_token = uq_pipe_across_tokens(seq_tokens_selection, emb_model=emb_model, question = example['question'], gen_ids = gen_ids, tokenizer_llm=llm.tokenizer, mode='selection')     
        vnes_token_disp, vnes_token_token_disp, vnes_token_add_combined_disp, vnes_token_multpl_combined_disp, vnes_token_deltas_disp, ln_probs_token, raos_token_disp, conflicts_token_disp, std_emb_token_disp = uq_pipe_across_tokens(seq_tokens_selection, emb_model=emb_model_deltas, question = example['question'], gen_ids = gen_ids, tokenizer_llm=llm.tokenizer, mode='selection')     

        vnes_word, vnes_word_ct, vnes_word_word, vnes_word_combined, vnes_word_proj, vnes_word_deltas, ln_probs_word, raos_word, conflicts_word, conflicts_word_ct, std_emb_word = uq_pipe_across_words(seq_words_selection, emb_model=emb_model, mode='selection')
        vnes_word_disp, vnes_word_ct_disp, vnes_word_word_disp, vnes_word_combined_disp, vnes_word_proj_disp, vnes_word_deltas_disp, ln_probs_word_disp, raos_word_disp, conflicts_word_disp, conflicts_word_ct_disp, std_emb_word_disp = uq_pipe_across_words(seq_words_selection, emb_model=emb_model_deltas, mode='selection')

        uqs_selection.append({'question': example['question'], 
                    'gen_text' : element['generated_text'], 
                    'gen_words': generated_words,
                    'gen_tokens' : generated_tokens,
                    'ses_word_to' : ses_words_to,
                    'ses_word_to_w' : ses_words_to_w,
                    'ses_tokens_to' : ses_tokens_to,
                    'ses_tokens_to_w' : ses_tokens_to_w,
                    'vnes_token' : vnes_token, 
                    'vnes_token_token' : vnes_token_token, 
                    'vnes_token_add_combined' : vnes_token_add_combined, 
                    'vnes_token_multpl_combined' : vnes_token_multpl_combined,
                    'vnes_token_deltas': vnes_token_deltas,
                    'vnes_word_emb' : vnes_word,
                    'vnes_word_emb_ct' : vnes_word_ct, 
                    'vnes_word_word' : vnes_word_word,
                    'vnes_word_add_combined' : vnes_word_proj, 
                    'vnes_word_multpl_combined' : vnes_word_combined,
                    'vnes_word_deltas' : vnes_word_deltas, 
                    'raos_token' : raos_token, 
                    'raos_word' : raos_word,
                    'conflicts_word': conflicts_word, 
                    'conflicts_word_ct' : conflicts_word_ct, 
                    'conflicts_token' : conflicts_token,
                    # --------------------------------- dispersion 
                    'vnes_token_disp' : vnes_token_disp, 
                    'vnes_token_token_disp' : vnes_token_token_disp, 
                    'vnes_token_add_combined_disp' : vnes_token_add_combined_disp, 
                    'vnes_token_multpl_combined_disp' : vnes_token_multpl_combined_disp,
                    'vnes_token_deltas_disp': vnes_token_deltas_disp,
                    'vnes_word_emb_disp' : vnes_word_disp,
                    'vnes_word_emb_ct_disp' : vnes_word_ct_disp, 
                    'vnes_word_word_disp' : vnes_word_word_disp,
                    'vnes_word_add_combined_disp' : vnes_word_proj_disp, 
                    'vnes_word_multpl_combined_disp' : vnes_word_combined_disp,
                    'vnes_word_deltas_disp' : vnes_word_deltas_disp, 
                    'raos_token_disp' : raos_token_disp, 
                    'raos_word_disp' : raos_word_disp,
                    'conflicts_word_disp': conflicts_word_disp, 
                    'conflicts_word_ct_disp' : conflicts_word_ct_disp, 
                    'conflicts_token_disp' : conflicts_token_disp,
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
    #save(uqs, f'{exp_name}_{ds_name}_uqs_all-MiniLM-L6-v2_weighted-2.pkl')
    save(uqs_sampled, f'{exp_name}_{ds_name}_uqs_sampled_100.pkl')
    save(uqs_selection, f'{exp_name}_{ds_name}_uqs_selection_100.pkl')

    
    
        
if __name__ == '__main__':
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    logging.info('STARTING `compute_uncertainty`!')
    main(args)
    logging.info('FINISHED `compute_uncertainty`!')