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
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, paired_cosine_distances, polynomial_kernel, cosine_similarity, laplacian_kernel
from utils.utils import get_parser, construct_prompt, save, get_metric, setup_logger, load
from data.utils import load_ds
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import numpy as np

#  --emb_model_id=models_peft/all-MiniLM-L6-v2-peft/final


setup_logger()

def se_pipe_across_tokens(question, seq_tokens, ellm, mode = 'adapted'): 
    cluster_ids_across_steps, topic_ids_across_steps = generate_semantic_subsequence_ids(seq_tokens=seq_tokens, question = question, ellm=ellm, mode=mode)
    entropies_ss = compute_se_across_subsequences(cluster_ids_across_steps=cluster_ids_across_steps, seq_tokens=seq_tokens, mode = 'complete') 
    entropies_to = compute_se_across_subsequences(cluster_ids_across_steps=cluster_ids_across_steps, seq_tokens=seq_tokens, mode = 'subsequ')  
    entropies_ss_weighted = compute_se_across_subsequences(cluster_ids_across_steps=cluster_ids_across_steps, seq_tokens=seq_tokens, mode = 'complete', topics = topic_ids_across_steps) 
    entropies_to_weighted = compute_se_across_subsequences(cluster_ids_across_steps=cluster_ids_across_steps, seq_tokens=seq_tokens, mode = 'subsequ', topics = topic_ids_across_steps)    
  
    return entropies_ss, entropies_to, entropies_ss_weighted, entropies_to_weighted 


def se_pipe_across_words(question, seq_words, ellm, mode = 'adapted'):   
    cluster_ids_across_steps, topic_ids_across_steps = generate_semantic_subsequence_ids(seq_tokens=seq_words, question = question, ellm=ellm, mode=mode)
    entropies_ss = compute_se_across_subsequences(cluster_ids_across_steps=cluster_ids_across_steps, seq_tokens=seq_words, mode = 'complete') 
    entropies_to = compute_se_across_subsequences(cluster_ids_across_steps=cluster_ids_across_steps, seq_tokens=seq_words, mode = 'subsequ')
    entropies_ss_weighted = compute_se_across_subsequences(cluster_ids_across_steps=cluster_ids_across_steps, seq_tokens=seq_words, mode = 'complete', topics = topic_ids_across_steps) 
    entropies_to_weighted = compute_se_across_subsequences(cluster_ids_across_steps=cluster_ids_across_steps, seq_tokens=seq_words, mode = 'subsequ', topics = topic_ids_across_steps)    
    return entropies_ss, entropies_to, entropies_ss_weighted, entropies_to_weighted 


def uq_pipe_across_tokens(seq_tokens, emb_model, emb_model_deltas, question, gen_ids, tokenizer_llm, tokenizer_emb):
    vnes = []
    vnes_tokens = []
    vnes_add_combined = []
    vnes_multpl_combined = []
    ln_probs = []
    raos = []
    conflicts = []
    for s_index, s in enumerate(seq_tokens): 
        ln_prob = - np.log(s['current_prob'])
        ln_probs.append(ln_prob)
        # Attach Question to generated prefix
        ems = emb_model.encode([question + ' ' + s for s in s['alternative_sequence_decoded']], normalize_embeddings=True) # 10, 384
        #ems_deltas = emb_model_deltas.encode([question + ' ' + s for s in s['alternative_sequence_decoded']], normalize_embeddings=True)
        ems_token = emb_model_deltas.encode(s['alternative_tokens_str'], normalize_embeddings=True)
        #prev_seq = question + tokenizer_llm.decode(gen_ids[:s_index], skip_special_tokens=True)
        #prev_ems = emb_model_deltas.encode(prev_seq, normalize_embeddings=True)
        #inputs = tokenizer_emb([question + ' ' + s for s in s['alternative_sequence_decoded']], return_tensors="pt", padding=True).to('cuda')
        #outputs = emb_model(input = inputs)#, output_hidden_states=True)
        #token_embs = outputs.token_embeddings
        
        #last_token_indices = outputs.attention_mask.sum(dim=1) - 1  # [batch]
        #last_valid_embs = token_embs[torch.arange(len(last_token_indices)),  # batch dimension (0..B-1)
        #                             last_token_indices-1                      # different token index per batch
        #                             ].detach().cpu().numpy()
        
        #embedding_deltas = prev_ems - ems_deltas
        #embedding_deltas = embedding_deltas / (np.linalg.norm(embedding_deltas, axis=1, keepdims=True) + 1e-12)
        
        #ke_deltas = kernel_noise(embedding_deltas, kernel=lambda x, y: cosine_similarity(x, y))
        
        
        #vne_deltas = vne(embedding_deltas) #, kernel=lambda x, y: cosine_similarity(x, y))
        vne_emb = vne(ems) #, kernel=lambda x, y: cosine_similarity(x, y))
        vne_tokens = vne(ems_token) #, kernel=lambda x, y: cosine_similarity(x, y))
        vne_add_combined = vne(Y = ems, Y2 = ems_token, combination_mode = "additive") #, kernel=lambda x, y: cosine_similarity(x, y))
        vne_multpl_combined = vne(ems, Y2 = ems_token, combination_mode = "multiplicative")
        rao_emb = rao_entropy(Y = ems, probs = s['alternative_token_probs'])
        conflict = avg_conflict(Y = ems, probs = s['alternative_token_probs'])
        
        vnes_tokens.append(vne_tokens)
        vnes.append(vne_emb)
        vnes_add_combined.append(vne_add_combined)
        vnes_multpl_combined.append(vne_multpl_combined)
        raos.append(rao_emb)
        conflicts.append(conflict)
        
    return vnes, vnes_tokens, vnes_add_combined, vnes_multpl_combined, ln_probs, raos, conflicts


def uq_pipe_across_words(seq_words, emb_model, emb_model_deltas):
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
    
    
    nlp = spacy.load("en_core_web_sm")
    for s_index, s in enumerate(seq_words): 
        previous_seq = s['prev_seq']
        alternative_sequences = s['alternative_sequence_decoded']
        
        ln_prob = - np.log(s['current_prob'])
        ln_probs.append(ln_prob)
        
        words = []
        for alternative in alternative_sequences:
            word = alternative[len(previous_seq):].lstrip()
            words.append(word)
        
        emb_model.eval()
        emb_model_deltas.eval()
        #with torch.no_grad():
        previous_emb = emb_model_deltas.encode(previous_seq, normalize_embeddings = True)
        device = next(emb_model[0].auto_model.parameters()).device
        # inputs = emb_model.tokenizer(alternative_sequences, return_tensors="pt", padding=True).to(device)            
        # # current_seq_tokens = emb_model_embs.tokenizer(current_sequence, return_tensors="pt", padding=True).to(device)
        # alternative_sequences_emb = emb_model(inputs)['sentence_embedding']
        # alternative_sequences_deltas = emb_model_deltas(inputs)['sentence_embedding']
        alternative_sequences_emb = emb_model.encode(alternative_sequences, normalize_embeddings = True)
        alternative_sequences_deltas = emb_model_deltas.encode(alternative_sequences, normalize_embeddings = True)
        words_emb = emb_model_deltas.encode(words, normalize_embeddings = True)       
        delta_embs = previous_emb - alternative_sequences_deltas
        delta_embs = delta_embs / (np.linalg.norm(delta_embs, axis=1, keepdims=True) + 1e-12)        
        
        #if consider_types: 
            # go through sequences, do pos_tagging, select last word
        word_pos = []
        for seq in alternative_sequences: 
            doc = nlp(seq)
            word_pos.append(doc[-1].pos_)
        word_pos = np.array(word_pos)
        mask_pos = (word_pos[:, None] != word_pos).astype(int)
        # special_mask = np.isin(word_pos, ["PUNCT", "SPACE", "SYM"])
        # mask[i, j] = 1 if either i or j is PUNCT/SPACE, else 0
        # mask_pos = (special_mask[:, None] | special_mask[None, :]).astype(int)
        
        print(mask_pos)
        
        if len(alternative_sequences) == 1:
            vne_emb, vne_ct, vne_word, vne_combined, rao_emb, conflict, conflict_ct = 0, 0, 0, 0, 0, 0, 0
        else:
            vne_emb = vne(Y = alternative_sequences_emb, center = True)#, kernel=lambda x, y: cosine_similarity(x, y))
            vne_ct = vne(Y = alternative_sequences_emb, type_mask=mask_pos, center = True)#, kernel=lambda x, y: cosine_similarity(x, y))
            vne_word = vne(Y = words_emb, center = True) #, kernel=lambda x, y: cosine_similarity(x, y))#, kernel=lambda x, y: cosine_similarity(x, y), type_mask=mask_pos)
            vne_combined = vne(Y = alternative_sequences_emb, Y2 = words_emb, center = True, combination_mode = "multiplicative") #, kernel=lambda x, y: cosine_similarity(x, y)) #, kernel=lambda x, y: cosine_similarity(x, y), type_mask=mask_pos)
            # vne_proj = vne(Y = alternative_sequences_emb, word_pos = word_pos, kernel=lambda x, y: cosine_similarity(x, y))
            vne_add =  vne(Y = alternative_sequences_emb, Y2 = words_emb, center = True, combination_mode = "additive")
            rao_emb = rao_entropy(Y = alternative_sequences_emb, probs = s['alternative_token_probs']) 
            conflict = avg_conflict(Y = alternative_sequences_emb, probs = s['alternative_token_probs']) 
            conflict_ct = avg_conflict(Y = alternative_sequences_emb, probs = s['alternative_token_probs'], type_mask = mask_pos) 
            
            if conflict_ct != conflict: 
                print("Different conflicts: ", conflict, conflict_ct)
        
        vnes.append(vne_emb)
        vnes_ct.append(vne_ct)
        vnes_word.append(vne_word)
        vnes_combined.append(vne_combined)
        vnes_proj.append(vne_add)
        raos.append(rao_emb)
        conflicts.append(conflict)
        conflicts_ct.append(conflict_ct)
            
    return vnes, vnes_ct, vnes_word, vnes_combined, vnes_proj, ln_probs, raos, conflicts, conflicts_ct          
        
        
def main(args): 
    model_id = args.model_id
    emb_model_id = args.emb_model_id
    emb_model_id_deltas = args.emb_model_id_deltas
    ellm_model_id = args.ellm_model_id
    exp_name = args.exp_name
    ds_name = args.dataset
    consider_types = args.consider_types
    
    generations = load(f'results_final/{exp_name}_{ds_name}_generations.pkl')
    logging.info('Dataset loaded!')
    logging.info(type(generations))
    uqs = []    
    llm = LLM(model_id=model_id)
    if model_id != ellm_model_id:
        if "nli" in ellm_model_id: 
            ellm = NLI(model_id=ellm_model_id)
        else:  
            ellm = LLM(model_id=ellm_model_id, storage_type='open_ai_api')
    emb_model = SentenceTransformer(emb_model_id) #SentenceTransformer("all-MiniLM-L6-v2")  
    emb_model.tokenizer.truncation_side = "left"  
    if emb_model_id != emb_model_id_deltas:
       emb_model_deltas = SentenceTransformer(emb_model_id_deltas)
       emb_model_deltas.tokenizer.truncation_side = "left" 
    
    tokenizer_emb = emb_model.tokenizer
    
    for e, element in enumerate(generations): 
        print("in element ------------------------------------------------------------------------------------------", e)
        logging.info(element['generated_text'])
        example = element['example']
        gen_ids = element['gen_ids']
        seq_tokens = element['seq_tokens']
        sampled_tokens = element['sampled_tokens']
        seq_words, generated_words = generate_word_subsequences(seq_tokens, element['generated_text'], example['question'], gen_ids, llm.tokenizer)
        
        if model_id != ellm_model_id: 
            #try:
            ses_words, ses_words_to, ses_words_w, ses_words_to_w = se_pipe_across_words(example['question'], seq_words, ellm, mode='adapted')
            ses_tokens, ses_tokens_to, ses_tokens_w, ses_tokens_to_w = se_pipe_across_tokens(example['question'], seq_tokens, ellm, mode='adapted')
            #except Exception as e:
            #    print('error in 212')
            #    print(e)
            #    logging.info(e)
            #    ses_words = None
            #    ses_tokens = None
                
        else: 
            try:
                ses_words, ses_words_to, ses_words_w, ses_words_to_w = se_pipe_across_words(example['question'], seq_words, llm)
                ses_tokens, ses_tokens_to, ses_tokens_w, ses_tokens_to_w  = se_pipe_across_tokens(example['question'], seq_tokens, llm)
            except Exception as e:
                print('error in 2123')
                print(e)
                logging.info(e)
                ses_words = None
                ses_tokens = None
                logging.info(e)
                ses_words = None
                ses_tokens = None
        
        #try:
        if emb_model_id != emb_model_id_deltas:
            vnes_token, vnes_token_token, vnes_token_add_combined, vnes_token_multpl_combined, ln_probs_token, raos_token, conflicts_token = uq_pipe_across_tokens(seq_tokens, emb_model=emb_model, emb_model_deltas=emb_model_deltas, question = example['question'], gen_ids = gen_ids, tokenizer_llm=llm.tokenizer, tokenizer_emb=tokenizer_emb)
        else: 
            vnes_token, vnes_token_token, vnes_token_add_combined, vnes_token_multpl_combined, ln_probs_token, raos_token, conflicts_token = uq_pipe_across_tokens(seq_tokens, emb_model=emb_model, emb_model_deltas=emb_model, question = example['question'], gen_ids = gen_ids, tokenizer_llm=llm.tokenizer, tokenizer_emb=tokenizer_emb)     
    # except Exception as e:
        #     logging.info(e)
        #     print('error in 239')
        #     print(e)
        #     logging.info(e)
        #     vnes_token, vnes_token_token, vnes_token_add_combined, vnes_token_multpl_combined, ln_probs_token, raos_token, conflicts_token = None, None, None, None, None, None, None 
        
        #try:
        if emb_model_id != emb_model_id_deltas:
            vnes_word, vnes_word_ct, vnes_word_word, vnes_word_combined, vnes_word_proj, ln_probs_word, raos_word, conflicts_word, conflicts_word_ct = uq_pipe_across_words(seq_words, emb_model=emb_model, emb_model_deltas=emb_model_deltas)
        else: 
            vnes_word, vnes_word_ct, vnes_word_word, vnes_word_combined, vnes_word_proj, ln_probs_word, raos_word, conflicts_word, conflicts_word_ct = uq_pipe_across_words(seq_words, emb_model=emb_model, emb_model_deltas=emb_model)
    #except Exception as e:
        #    print('error in 254')
        #    print(e)
        #    vnes_word, vnes_word_ct, vnes_word_word, vnes_word_combined, vnes_word_proj, ln_probs_word, raos_word, conflicts_word, conflicts_word_ct = None, None, None, None, None, None, None, None, None
        
        uqs.append({'question': example['question'], 
                    'gen_text' : element['generated_text'], 
                    'gen_words': generated_words,
                    'gen_ids' : gen_ids, 
                    'ses_word' : ses_words,
                    'ses_word_w' : ses_words_w,
                    'ses_word_to' : ses_words_to,
                    'ses_word_to_w' : ses_words_to_w,
                    'ses_tokens' : ses_tokens,
                    'ses_tokens_to' : ses_tokens_to,
                    'ses_tokens_w' : ses_tokens_w,
                    'ses_tokens_to_w' : ses_tokens_to_w,
                    'vnes_token' : vnes_token, 
                    'vnes_token_token' : vnes_token_token, 
                    'vnes_token_add_combined' : vnes_token_add_combined, 
                    'vnes_token_multpl_combined' : vnes_token_multpl_combined,
                    'vnes_word_emb' : vnes_word,
                    'vnes_word_emb_ct' : vnes_word_ct, 
                    'vnes_word_word' : vnes_word_word,
                    'vnes_word_add_combined' : vnes_word_proj, 
                    'vnes_word_multpl_combined' : vnes_word_combined,
                    'raos_token' : raos_token, 
                    'raos_word' : raos_word,
                    'conflicts_word': conflicts_word, 
                    'conflicts_word_ct' : conflicts_word_ct, 
                    'conflicts_token' : conflicts_token,
                    'ln_probs_word' : ln_probs_word,
                    'ln_probs_token' : ln_probs_token,
                    'true_answer' : example['answer'], 
                    'sampled_tokens' : sampled_tokens,
                    'sampled_words' : seq_words
                    })
    
    if model_id != ellm_model_id:
        del ellm
    del llm
    save(uqs, f'{exp_name}_{ds_name}_uqs_all-MiniLM-L6-v2_rbf-15-test.pkl')
        
if __name__ == '__main__':
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    logging.info('STARTING `compute_uncertainty`!')
    main(args)
    logging.info('FINISHED `compute_uncertainty`!')