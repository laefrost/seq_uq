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
from utils.subsequences import generate_subsequences, remove_subsequences, generate_word_subsequences
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, paired_cosine_distances, polynomial_kernel, cosine_similarity
from utils.utils import get_parser, construct_prompt, save, get_metric, setup_logger, load
from data.utils import load_ds
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

#  --emb_model_id=models_peft/all-MiniLM-L6-v2-peft/final


setup_logger()

def se_pipe_across_tokens(question, seq_tokens, ellm, mode = 'adapted'): 
    cluster_ids_across_steps, cluster_weights_across_steps = generate_semantic_subsequence_ids(seq_tokens=seq_tokens, question = question, ellm=ellm, mode=mode)
    print('weights ----------', cluster_weights_across_steps)
    entropies_ss = compute_se_across_subsequences(cluster_ids_across_steps=cluster_ids_across_steps, seq_tokens=seq_tokens, mode = 'complete') 
    entropies_to = compute_se_across_subsequences(cluster_ids_across_steps=cluster_ids_across_steps, seq_tokens=seq_tokens, mode = 'subsequ')  
    entropies_ss_weighted = compute_se_across_subsequences(cluster_ids_across_steps=cluster_ids_across_steps, seq_tokens=seq_tokens, mode = 'complete', weights = cluster_weights_across_steps) 
    entropies_to_weighted = compute_se_across_subsequences(cluster_ids_across_steps=cluster_ids_across_steps, seq_tokens=seq_tokens, mode = 'subsequ', weights = cluster_weights_across_steps)    
  
    return entropies_ss, entropies_to, entropies_ss_weighted, entropies_to_weighted # {'gen_text' : generated_text, 'entropies' : entropies, 'gen_ids' : gen_ids, 'true_answer' : example['answer']}#['aliases']}


def se_pipe_across_words(question, seq_words, ellm, mode = 'adapted'):   
    cluster_ids_across_steps, cluster_weights_across_steps = generate_semantic_subsequence_ids(seq_tokens=seq_words, question = question, ellm=ellm, mode=mode)
    print('weights ----------', cluster_weights_across_steps)
    entropies_ss = compute_se_across_subsequences(cluster_ids_across_steps=cluster_ids_across_steps, seq_tokens=seq_words, mode = 'complete') 
    entropies_to = compute_se_across_subsequences(cluster_ids_across_steps=cluster_ids_across_steps, seq_tokens=seq_words, mode = 'subsequ')
    entropies_ss_weighted = compute_se_across_subsequences(cluster_ids_across_steps=cluster_ids_across_steps, seq_tokens=seq_words, mode = 'complete', weights = cluster_weights_across_steps) 
    entropies_to_weighted = compute_se_across_subsequences(cluster_ids_across_steps=cluster_ids_across_steps, seq_tokens=seq_words, mode = 'subsequ', weights = cluster_weights_across_steps)    
    return entropies_ss, entropies_to, entropies_ss_weighted, entropies_to_weighted # {'gen_text' : generated_text, 'entropies' : entropies, 'gen_ids' : gen_ids, 'true_answer' : example['answer']}#['aliases']}


def uq_pipe_across_tokens(seq_tokens, emb_model, emb_model_deltas, question, gen_ids, tokenizer_llm, tokenizer_emb):
    kes_og = []
    kes_sum = []
    kes_word = []
    kes_deltas = []
    kes_tokens = []
    vnes = []
    vnes_delta = []
    vnes_tokens = []
    vnes_combined = []
    for s_index, s in enumerate(seq_tokens): 
        ems = emb_model.encode([question + ' ' + s for s in s['alternative_sequence_decoded']], normalize_embeddings=True) # 10, 384
        ems_deltas = emb_model_deltas.encode([question + ' ' + s for s in s['alternative_sequence_decoded']], normalize_embeddings=True)
        ems_word = emb_model.encode(s['alternative_tokens_str'], normalize_embeddings=True)
        prev_seq = question + tokenizer_llm.decode(gen_ids[:s_index], skip_special_tokens=True)
        prev_ems = emb_model_deltas.encode(prev_seq, normalize_embeddings=True)
        inputs = tokenizer_emb([question + ' ' + s for s in s['alternative_sequence_decoded']], return_tensors="pt", truncation=True, padding=True, max_length=512).to('cuda')
        outputs = emb_model(input = inputs)#, output_hidden_states=True)
        token_embs = outputs.token_embeddings
        
        last_token_indices = outputs.attention_mask.sum(dim=1) - 1  # [batch]
        last_valid_embs = token_embs[torch.arange(len(last_token_indices)),  # batch dimension (0..B-1)
                                     last_token_indices-1                      # different token index per batch
                                     ].detach().cpu().numpy()
        
        ke_1 = kernel_noise(ems, kernel=lambda x, y: cosine_similarity(x, y))
        ke_2 = kernel_noise(ems + ems_word, kernel=lambda x, y: cosine_similarity(x, y))
        ke_3 = kernel_noise(ems_word, kernel=lambda x, y: cosine_similarity(x, y))
        ke_4 = kernel_noise(last_valid_embs, kernel=lambda x, y: cosine_similarity(x, y))
        
        embedding_deltas = prev_ems - ems_deltas
        embedding_deltas = embedding_deltas / (np.linalg.norm(embedding_deltas, axis=1, keepdims=True) + 1e-12)
        
        ke_deltas = kernel_noise(embedding_deltas, kernel=lambda x, y: cosine_similarity(x, y))
        vne_deltas = vne(embedding_deltas, kernel=lambda x, y: cosine_similarity(x, y))
        vne_emb = vne(ems, kernel=lambda x, y: cosine_similarity(x, y))
        vne_tokens = vne(last_valid_embs, kernel=lambda x, y: cosine_similarity(x, y))
        vne_combined_tokens = vne(0.5 * ems + 0.5 * embedding_deltas, kernel=lambda x, y: cosine_similarity(x, y))
        kes_og.append(ke_1)
        kes_sum.append(ke_2)
        kes_word.append(ke_3)
        kes_tokens.append(ke_4)
        kes_deltas.append(ke_deltas)
        vnes_delta.append(vne_deltas)
        vnes.append(vne_emb)
        vnes_tokens.append(vne_tokens)
        vnes_combined.append(vne_combined_tokens)
        
    return kes_og, kes_sum, kes_word, kes_deltas, kes_tokens, vnes, vnes_delta, vnes_tokens, vnes_combined


def uq_pipe_across_words(seq_words, emb_model, emb_model_deltas, consider_types = False):
    l = 1
    kes_emb = []
    kes_delta = []
    kes_grad = []
    vnes_grad = []
    vnes = []
    vnes_delta = []
    vnes_combined = []
    
    if consider_types: 
        nlp = spacy.load("en_core_web_sm")
    for s_index, s in enumerate(seq_words): 
        previous_seq = s['prev_seq']
        alternative_sequences = s['alternative_sequence_decoded']
        current_sequence = s['current_seq']
        
        emb_model.eval()
        emb_model_deltas.eval()
        with torch.no_grad():
            previous_emb = emb_model_deltas.encode(previous_seq)
        

        # Step 1: get gradient of cosine loss w.r.t embeddings from embedding model for current sequence
        #with torch.enable_grad():
            # latest_grads = [None]
            # latest_embeddings = [None]
            
            # def forward_hook(module, input, output):
            #     latest_embeddings[0] = output.detach()
            #     output.register_hook(lambda grad: latest_grads.__setitem__(0, grad))
            
            # handle = emb_model[0].auto_model.embeddings.word_embeddings.register_forward_hook(forward_hook)

            device = next(emb_model[0].auto_model.parameters()).device
            
            inputs = emb_model.tokenizer(alternative_sequences, return_tensors="pt", padding=True, truncation=True).to(device)            
            # current_seq_tokens = emb_model_embs.tokenizer(current_sequence, return_tensors="pt", padding=True, truncation=True).to(device)
            
            alternative_sequences_emb = emb_model(inputs)['sentence_embedding']
            alternative_sequences_deltas = emb_model_deltas(inputs)['sentence_embedding']
            print(alternative_sequences_emb.shape)
            
            #loss_fn = F.l1_loss(alternative_sequences_emb, -alternative_sequences_emb)
            #loss_fn.backward()

        #handle.remove()

        #del loss_fn
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # with torch.no_grad():  
        #     e = latest_grads[0].shape[1]-1
        #     mask = inputs['attention_mask'].unsqueeze(-1)  # shape (batch, seq_len, 1)
        #     masked_grads = latest_grads[0] * mask
        #     grad = masked_grads[:, l:e ,:]
        #     grad_copy = grad.detach().clone()
        #     l = current_seq_tokens['input_ids'][0].shape[0]-2
        #     grad_mean = torch.mean(grad_copy, dim=1).to('cpu')


        #delta_embs = previous_emb - alternative_sequences_emb.detach().cpu().numpy()
        delta_embs = previous_emb - alternative_sequences_deltas.detach().cpu().numpy()
        delta_embs = delta_embs / (np.linalg.norm(delta_embs, axis=1, keepdims=True) + 1e-12)
        
        mask_pos = None
        
        if consider_types: 
            # go through sequences, do pos_tagging, select last word
            word_pos = []
            for seq in alternative_sequences: 
                doc = nlp(seq)
                word_pos.append(doc[-1].pos_)
            
            word_pos = np.array(word_pos)
            # mask_pos = (word_pos[:, None] != word_pos).astype(int)
            special_mask = np.isin(word_pos, ["PUNCT", "SPACE", "SYM"])

            # mask[i, j] = 1 if either i or j is PUNCT/SPACE, else 0
            mask_pos = (special_mask[:, None] | special_mask[None, :]).astype(int)
            
        if len(alternative_sequences) == 1: 
            ke, ke_delta = -1, -1
        else:
            ke = kernel_noise(Y = alternative_sequences_emb.detach().cpu().numpy(), kernel=lambda x, y: cosine_similarity(x, y), type_mask=mask_pos)
            ke_delta = kernel_noise(Y = delta_embs, kernel=lambda x, y: cosine_similarity(x, y), type_mask=mask_pos)
        # ke_grad = kernel_noise(Y = grad_mean.detach().cpu().numpy(), kernel =lambda x, y: cosine_similarity(x, y), type_mask=mask_pos)
        ke_grad = None
        kes_emb.append(ke)
        kes_delta.append(ke_delta)
        kes_grad.append(ke_grad)
        
        if len(alternative_sequences) == 1:
            vne_delta, vne_emb, vne_combined = 0, 0, 0
        else:
            vne_emb = vne(Y = alternative_sequences_emb.detach().cpu().numpy(), kernel=lambda x, y: cosine_similarity(x, y), type_mask=mask_pos)
            vne_delta = vne(Y = delta_embs, kernel=lambda x, y: cosine_similarity(x, y), type_mask=mask_pos)
            vne_combined = vne(Y = 0.5 * delta_embs + 0.5 * alternative_sequences_emb.detach().cpu().numpy(), kernel=lambda x, y: cosine_similarity(x, y), type_mask=mask_pos)
        # vne_grad = vne(Y = grad_mean.detach().cpu().numpy(), kernel=lambda x, y: cosine_similarity(x, y), type_mask=mask_pos)
        vne_grad = None
        vnes.append(vne_emb)
        vnes_delta.append(vne_delta)
        vnes_grad.append(vne_grad)
        vnes_combined.append(vne_combined)
            
    return kes_emb, kes_delta, kes_grad, vnes, vnes_delta, vnes_grad, vnes_combined        



# def uq_pipe_across_words(instance, emb_model, tokenizer, consider_types = False):
#     skipped = 0 
#     # pattern = r"\([0-9]+(?:[-–][0-9]+)*\)|[0-9]+(?:[.,-][0-9]+)*|[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)*|[.,;?!:]"
#     pattern = r"\([0-9]+(?:[-–][0-9]+)*\)|[0-9]+(?:[.,-][0-9]+)*|[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)*|[.,;?!:]|\n|\(|\)|</s>|\'|\"|\?|\!|\`|\´|\-|-"
    
#     gen_text = tokenizer.decode(instance['gen_ids'], skip_special_tokens=True)
#     generated_words = re.findall(pattern, instance['generated_text'])
#     question = instance['example']['question']
#     tokens = instance['gen_ids']
#     l = 1
#     kes_emb = []
#     kes_delta = []
#     kes_grad = []
#     vnes_grad = []
#     vnes = []
#     vnes_delta = []
#     previous_seq = question

#     if consider_types: 
#         nlp = spacy.load("en_core_web_sm")

#     token_idx = 0
#     skipped_words = 0

#     for w, word in enumerate(generated_words): 
#         torch.cuda.empty_cache()
#         if skipped_words > 0: 
#             skipped_words = skipped_words - 1
#             skipped = skipped - 1 
#             continue

#         emb_model.zero_grad(set_to_none=True)        
        
        
#         word_tokens = [tokens[token_idx]]
#         decoded_tokens = tokenizer.decode(word_tokens)
#         #print('tokenized tokens:', decoded_tokens, len(decoded_tokens),'word', word, len(word))
#         if len(decoded_tokens) < len(word):   
#             #print('decoded tokens smaller than word')
#             while decoded_tokens != word:
#                 #print('tokenized tokens:', decoded_tokens, 'word', word)
#                 token_idx = token_idx + 1 
#                 word_tokens.append(tokens[token_idx])
#                 decoded_tokens = tokenizer.decode(word_tokens)
#         else:
#             # print('either word or bigger')
#             skipped_words = 0
#             while decoded_tokens != word and skipped_words < len(generated_words):
#                     # print('tokenized tokens:', decoded_tokens,'word', word)
#                     skipped_words = skipped_words + 1 
#                     #word_tokens.append(tokens[token_idx])
#                     word = word + generated_words[w+1]
        
#         token_idx = token_idx + 1 
#         # corresponding_tokens = []
#         # for i, t in enumerate(corresponding_tokens_raw):
#         #     if t == '▁' and i + 1 < len(corresponding_tokens_raw) and corresponding_tokens_raw[i + 1] == '<0x0A>':
#         #         continue  # skip the dummy space before newline
#         #     corresponding_tokens.append(t)
#         # print('finallllll:', decoded_tokens, len(decoded_tokens),'word', word, len(word))
#         emb_model.eval()
#         with torch.no_grad():
#             if len(word_tokens) > 1: 
#                 alternative_sequences = []
#                 for i, token in enumerate(word_tokens):
#                     alternative_sequences.extend([question + ' ' + t for t in instance['seq_tokens'][w + skipped + i]['s_decoded']])
                
#                 alternative_sequences = remove_subsequences(alternative_sequences)
#                 # Since we skipped the tokens belonging to the current word
#                 skipped = skipped + len(word_tokens) - 1
#             else: 
#                 # print('index looke at: ', w + skipped)
#                 alternative_sequences = [question + ' ' + t for t in instance['seq_tokens'][w + skipped]['s_decoded']] #test_case['seq_tokens'][w + skipped]['s_decoded']
#             previous_emb = emb_model.encode(previous_seq)
        

#         # Step 1: get gradient of cosine loss w.r.t embeddings from embedding model for current sequence
#         with torch.enable_grad():
#             latest_grads = [None]
#             latest_embeddings = [None]
            
#             def forward_hook(module, input, output):
#                 latest_embeddings[0] = output.detach()
#                 output.register_hook(lambda grad: latest_grads.__setitem__(0, grad))
            
#             handle = emb_model[0].auto_model.embeddings.word_embeddings.register_forward_hook(forward_hook)

#             device = next(emb_model[0].auto_model.parameters()).device
#             inputs = emb_model.tokenizer(alternative_sequences, return_tensors="pt", padding=True, truncation=True).to(device)
#             current_seq_tokens = emb_model.tokenizer(current_sequence, return_tensors="pt", padding=True, truncation=True).to(device)
            
#             alternative_sequences_emb = emb_model(inputs)['sentence_embedding']
            
#             loss_fn = F.l1_loss(alternative_sequences_emb, -alternative_sequences_emb)
#             loss_fn.backward()

#         handle.remove()

#         del loss_fn
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()

#         with torch.no_grad():  
#             e = latest_grads[0].shape[1]-1
#             mask = inputs['attention_mask'].unsqueeze(-1)  # shape (batch, seq_len, 1)
#             masked_grads = latest_grads[0] * mask
#             grad = masked_grads[:, l:e ,:]
#             grad_copy = grad.detach().clone()
#             l = current_seq_tokens['input_ids'][0].shape[0]-2
#             grad_mean = torch.mean(grad_copy, dim=1).to('cpu')


#         delta_embs = previous_emb - alternative_sequences_emb.detach().cpu().numpy()
#         delta_embs = delta_embs / (np.linalg.norm(delta_embs, axis=1, keepdims=True) + 1e-12)
        
#         mask_pos = None
        
#         if consider_types: 
#             # go through sequences, do pos_tagging, select last word
#             word_pos = []
#             for seq in alternative_sequences: 
#                 doc = nlp(seq)
#                 word_pos.append(doc[-1].pos_)
            
#             word_pos = np.array(word_pos)
#             # mask_pos = (word_pos[:, None] != word_pos).astype(int)
#             special_mask = np.isin(word_pos, ["PUNCT", "SPACE", "SYM"])

#             # mask[i, j] = 1 if either i or j is PUNCT/SPACE, else 0
#             mask_pos = (special_mask[:, None] | special_mask[None, :]).astype(int)
            
#         ke = kernel_noise(Y = alternative_sequences_emb.detach().cpu().numpy(), kernel=lambda x, y: cosine_similarity(x, y), type_mask=mask_pos)
#         ke_delta = kernel_noise(Y = delta_embs, kernel=lambda x, y: cosine_similarity(x, y), type_mask=mask_pos)
#         ke_grad = kernel_noise(Y = grad_mean.detach().cpu().numpy(), kernel =lambda x, y: cosine_similarity(x, y), type_mask=mask_pos)

#         kes_emb.append(ke)
#         kes_delta.append(ke_delta)
#         kes_grad.append(ke_grad)
        
#         vne_emb = vne(Y = alternative_sequences_emb.detach().cpu().numpy(), kernel=lambda x, y: cosine_similarity(x, y), type_mask=mask_pos)
#         vne_delta = vne(Y = delta_embs, kernel=lambda x, y: cosine_similarity(x, y), type_mask=mask_pos)
#         vne_grad = vne(Y = grad_mean.detach().cpu().numpy(), kernel=lambda x, y: cosine_similarity(x, y), type_mask=mask_pos)
        
#         vnes.append(vne_emb)
#         vnes_delta.append(vne_delta)
#         vnes_grad.append(vne_grad)
        
#         previous_seq = question + ' ' + current_sequence
    
#     return kes_emb, kes_delta, kes_grad, vnes, vnes_delta, vnes_grad            
        
        
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
    # Initialize model
    # TODO: Maybe use OpenAI or other models to create NLI call
    llm = LLM(model_id=model_id)
    if model_id != ellm_model_id:
        if "nli" in ellm_model_id: 
            ellm = NLI(model_id=ellm_model_id)
        else:  
            ellm = LLM(model_id=ellm_model_id, storage_type='open_ai_api')
    emb_model = SentenceTransformer(emb_model_id) #SentenceTransformer("all-MiniLM-L6-v2")    
    if emb_model_id != emb_model_id_deltas:
       emb_model_deltas = SentenceTransformer(emb_model_id_deltas) 
    #emb_model[0].auto_model = PeftModel.from_pretrained(emb_model[0].auto_model,
    #"models_peft/all-MiniLM-L6-v2-peft/final")
    # 2. Load your PEFT adapter on top
    #emb_model.load_adapter("models_peft/all-MiniLM-L6-v2-peft/final")
    tokenizer_emb = emb_model.tokenizer# AutoTokenizer.from_pretrained("sentence-transformers/" + emb_model_id)
    
    for e, element in enumerate(generations): 
        logging.info(element['generated_text'])
        example = element['example']
        gen_ids = element['gen_ids']
        seq_tokens = element['seq_tokens']
        sampled_tokens = element['sampled_tokens']
        seq_words, generated_words = generate_word_subsequences(seq_tokens, element['generated_text'], example['question'], gen_ids, llm.tokenizer)
        print('length words, tokens ',len(seq_tokens), len(seq_words), len(generated_words))
        print(generated_words)
        
        if model_id != ellm_model_id: 
            #try:
            ses_words, ses_words_to, ses_words_w, ses_words_to_w = se_pipe_across_words(example['question'], seq_words, ellm, mode='adapted')
            ses_tokens, ses_tokens_to, ses_tokens_w, ses_tokens_to_w = se_pipe_across_tokens(example['question'], seq_tokens, ellm, mode='adapted')
            #except Exception as e:
            #    print('in except 1')
            #    print("Error in token-level UQ:", e)
            #    ses_words = None
            #    ses_tokens = None
                
        else: 
            #try:
            ses_words, ses_words_to, ses_words_w, ses_words_to_w = se_pipe_across_words(example['question'], seq_words, llm)
            ses_tokens, ses_tokens_to, ses_tokens_w, ses_tokens_to_w  = se_pipe_across_tokens(example['question'], seq_tokens, llm)
            #except Exception as e:
            #    print('in except 1')
            #    print("Error in token-level UQ:", e)
        #ses_words = None
        #ses_tokens = None
        
        try:
            if emb_model_id != emb_model_id_deltas:
                pkes_token_emb, pkes_token_sum, pkes_token_word, pkes_token_deltas, pkes_token_token, vnes_token_emb, vnes_token_deltas, vnes_token_token, vnes_token_combined = uq_pipe_across_tokens(seq_tokens, emb_model=emb_model, emb_model_deltas=emb_model_deltas, question = example['question'], gen_ids = gen_ids, tokenizer_llm=llm.tokenizer, tokenizer_emb=tokenizer_emb)
            else: 
                pkes_token_emb, pkes_token_sum, pkes_token_word, pkes_token_deltas, pkes_token_token, vnes_token_emb, vnes_token_deltas, vnes_token_token, vnes_token_combined = uq_pipe_across_tokens(seq_tokens, emb_model=emb_model, emb_model_deltas=emb_model, question = example['question'], gen_ids = gen_ids, tokenizer_llm=llm.tokenizer, tokenizer_emb=tokenizer_emb)
                
        except Exception as e:
            print('in except 2')
            print("Error in token-level UQ:", e)
            pkes_token_emb, pkes_token_sum, pkes_token_word, pkes_token_deltas, pkes_token_token, vnes_token_emb, vnes_token_deltas, vnes_token_token, vnes_token_combined = None, None, None, None, None, None, None, None, None 
        
        try:
            if emb_model_id != emb_model_id_deltas:
                pkes_word_emb, pkes_word_deltas, pkes_word_grad, vnes_word_emb, vnes_word_deltas, vnes_word_grad, vnes_word_combined  = uq_pipe_across_words(seq_words, emb_model=emb_model, emb_model_deltas=emb_model_deltas, consider_types=consider_types)
            else: 
                pkes_word_emb, pkes_word_deltas, pkes_word_grad, vnes_word_emb, vnes_word_deltas, vnes_word_grad, vnes_word_combined  = uq_pipe_across_words(seq_words, emb_model=emb_model, emb_model_deltas=emb_model, consider_types=consider_types)
                
        except Exception as e:
            print('in except 3')
            print("Error in token-level UQ:", e)
            pkes_word_emb, pkes_word_deltas, pkes_word_grad, vnes_word_emb, vnes_word_deltas, vnes_word_grad, vnes_word_combined = None, None, None, None, None, None, None
        
        #{'gen_text' : generated_text, 'entropies' : entropies, 'gen_ids' : gen_ids, 'pkes': pkes, 'true_answer' : example['answer']}#['aliases']}
        uqs.append({'question': example['question'], 
                    'gen_text' : element['generated_text'], 
                    'gen_words': generated_words,
                    'gen_ids' : gen_ids, 
                    'ses_token' : ses_tokens, 
                    'ses_word' : ses_words,
                    'ses_token_w' : ses_tokens_w, 
                    'ses_word_w' : ses_words_w,
                    'ses_token_to' : ses_tokens_to, 
                    'ses_word_to' : ses_words_to,
                    'ses_token_to_w' : ses_tokens_to_w, 
                    'ses_word_to_' : ses_words_to_w,
                    'pkes_token_emb': pkes_token_emb, 
                    'pkes_token_sum' : pkes_token_sum, 
                    'pkes_token_word' : pkes_token_word, 
                    'pkes_token_deltas': pkes_token_deltas, 
                    'pkes_token_token' : pkes_token_token, 
                    'pkes_word_emb': pkes_word_emb, 
                    'pkes_word_deltas' : pkes_word_deltas, 
                    'pkes_word_grad' : pkes_word_grad,
                    'vnes_token_deltas' : vnes_token_deltas, 
                    'vnes_token_emb' : vnes_token_emb, 
                    'vnes_word_emb' : vnes_word_emb,
                    'vnes_word_deltas' : vnes_word_deltas, 
                    'vnes_word_grad' : vnes_word_grad,
                    'vnes_token_token' : vnes_token_token,
                    'vnes_word_combined': vnes_word_combined,
                    'vnes_token_combined' : vnes_token_combined,  
                    'true_answer' : example['answer'], 
                    'sampled_tokens' : sampled_tokens,
                    'sampled_words' : seq_words
                    #'acc' : element['acc'],
                    #'acc_tokens' : element['acc_tokens'],
                    #'acc_words' : element['acc_words']
                    })
    
    if model_id != ellm_model_id:
        del ellm
    del llm
    save(uqs, f'{exp_name}_{ds_name}_uqs_all-MiniLM-L6-v2.pkl')
        
if __name__ == '__main__':
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    logging.info('STARTING `compute_uncertainty`!')
    main(args)
    logging.info('FINISHED `compute_uncertainty`!')