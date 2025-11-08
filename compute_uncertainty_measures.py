import torch
import re
import torch.nn.functional as F
import spacy
import gc
import logging
import random
from sentence_transformers import SentenceTransformer

from models.models import * 
from uncertainty_metrics.se import * 
from uncertainty_metrics.pke import *
from utils.subsequences import generate_subsequences, remove_subsequences
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, paired_cosine_distances, polynomial_kernel, cosine_similarity
from utils.utils import get_parser, construct_prompt, save, get_metric, setup_logger, load
from data.utils import load_ds


setup_logger()

def se_pipe(question, seq_tokens, ellm, tokenizer): 
    cluster_ids_across_steps = generate_semantic_subsequence_ids(seq_tokens=seq_tokens, question = question, ellm=ellm, tokenizer=tokenizer)
    entropies = compute_se_across_subsequences(cluster_ids_across_steps=cluster_ids_across_steps, seq_tokens=seq_tokens)    
    return entropies # {'gen_text' : generated_text, 'entropies' : entropies, 'gen_ids' : gen_ids, 'true_answer' : example['answer']}#['aliases']}


def pke_pipe_across_tokens(seq_tokens, emb_model, question):
    kes_og = []
    kes_sum = []
    kes_word = []
    kes_deltas = []
    prev_seq = question
    for s in seq_tokens: 
        print(s['s_decoded'])
        ems = emb_model.encode([question + ' ' + s for s in s['s_decoded']], normalize_embeddings=True) # 10, 384
        ems_word = emb_model.encode(s['s_str'], normalize_embeddings=True)
        prev_ems = emb_model.encode(prev_seq)
        ke_1 = kernel_noise(ems, kernel=lambda x, y: cosine_similarity(x, y))
        ke_2 = kernel_noise(ems + ems_word, kernel=lambda x, y: cosine_similarity(x, y))
        ke_3 = kernel_noise(ems_word, kernel=lambda x, y: cosine_similarity(x, y))
        embedding_deltas = prev_ems - ems
        ke_deltas = kernel_noise(embedding_deltas, kernel=lambda x, y: cosine_similarity(x, y))
        kes_og.append(ke_1)
        kes_sum.append(ke_2)
        kes_word.append(ke_3)
        kes_deltas.append(ke_deltas)
    return kes_og, kes_sum, kes_word, kes_deltas



def pke_pipe_across_words(instance, emb_model, tokenizer, consider_types = False):
    skipped = 0 
    pattern = r"\([0-9]+(?:[-–][0-9]+)*\)|[0-9]+(?:[.,-][0-9]+)*|[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)*|[.,;?!:]"
    generated_words = re.findall(pattern, instance['generated_text'])
    question = instance['example']['question']
    l = 1
    kes_emb = []
    kes_delta = []
    kes_grad = []
    previous_seq = question
    
    if consider_types: 
        nlp = spacy.load("en_core_web_sm")

    for w, word in enumerate(generated_words): 
        torch.cuda.empty_cache()

        emb_model.zero_grad(set_to_none=True)        
        corresponding_tokens = tokenizer.tokenize(word)
        current_sequence = ' '.join(generated_words[:w+1])

        emb_model.eval()
        with torch.no_grad():
            if len(corresponding_tokens) > 1: 
                alternative_sequences = []
                for i, token in enumerate(corresponding_tokens):
                    alternative_sequences.extend([question + ' ' + t for t in instance['seq_tokens'][w + skipped + i]['s_decoded']])
                
                alternative_sequences = remove_subsequences(alternative_sequences)
                # Since we skipped the tokens belonging to the current word
                skipped = skipped + len(corresponding_tokens) - 1
            else: 
                alternative_sequences = [question + ' ' + t for t in instance['seq_tokens'][w + skipped]['s_decoded']] #test_case['seq_tokens'][w + skipped]['s_decoded']
            previous_emb = emb_model.encode(previous_seq)
        

        # Step 1: get gradient of cosine loss w.r.t embeddings from embedding model for current sequence
        with torch.enable_grad():
            latest_grads = [None]
            latest_embeddings = [None]
            
            def forward_hook(module, input, output):
                latest_embeddings[0] = output.detach()
                output.register_hook(lambda grad: latest_grads.__setitem__(0, grad))
            
            handle = emb_model[0].auto_model.embeddings.word_embeddings.register_forward_hook(forward_hook)

            device = next(emb_model[0].auto_model.parameters()).device
            inputs = emb_model.tokenizer(alternative_sequences, return_tensors="pt", padding=True, truncation=True).to(device)
            current_seq_tokens = emb_model.tokenizer(current_sequence, return_tensors="pt", padding=True, truncation=True).to(device)
            
            alternative_sequences_emb = emb_model(inputs)['sentence_embedding']
            
            loss_fn = F.l1_loss(alternative_sequences_emb, -alternative_sequences_emb)
            loss_fn.backward()

        handle.remove()

        del loss_fn
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        with torch.no_grad():  
            e = latest_grads[0].shape[1]-1
            mask = inputs['attention_mask'].unsqueeze(-1)  # shape (batch, seq_len, 1)
            masked_grads = latest_grads[0] * mask
            grad = masked_grads[:, l:e ,:]
            grad_copy = grad.detach().clone()
            l = current_seq_tokens['input_ids'][0].shape[0]-2
            grad_mean = torch.mean(grad_copy, dim=1).to('cpu')


        delta_embs = previous_emb - alternative_sequences_emb.detach().cpu().numpy()
        
        mask_pos = None
        
        if consider_types: 
            # go through sequences, do pos_tagging, select last word
            word_pos = []
            for seq in alternative_sequences: 
                doc = nlp(seq)
                word_pos.append(doc[-1].pos_)
            
            word_pos = np.array(word_pos)
            mask_pos = (word_pos[:, None] != word_pos).astype(int)
            
        ke = kernel_noise(Y = alternative_sequences_emb.detach().cpu().numpy(), kernel=lambda x, y: cosine_similarity(x, y), type_mask=mask_pos)
        ke_delta = kernel_noise(Y = delta_embs, kernel=lambda x, y: cosine_similarity(x, y), type_mask=mask_pos)
        ke_grad = kernel_noise(Y = grad_mean.detach().cpu().numpy(), kernel =lambda x, y: cosine_similarity(x, y), type_mask=mask_pos)

        kes_emb.append(ke)
        kes_delta.append(ke_delta)
        kes_grad.append(ke_grad)
        previous_seq = question + ' ' + current_sequence
    
    return kes_emb, kes_delta, kes_grad        
        
        
def main(args): 
    model_id = args.model_id
    exp_name = args.exp_name
    ds_name = args.dataset
    
    generations = load(f'{exp_name}_{ds_name}_generations.pkl')
    logging.info('Dataset loaded!')
    
    # Initialize model
    llm = LLM(model_id=model_id)
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    uqs = []
    for e, element in enumerate(generations): 
        print(element['generated_text'])
        example = element['example']
        gen_ids = element['gen_ids']
        seq_tokens = element['seq_tokens']
        sampled_tokens = element['sampled_tokens']
        
        # TODO: Hier einmal die OG und dann die angepasste Version nehmen
        # entropies = se_pipe(example['question'], seq_tokens, ellm, tokenizer)
        
        # Verschiedene Versionen von PKE
        pkes_token_emb, pkes_token_sum, pkes_token_word, pke_token_deltas = pke_pipe_across_tokens(seq_tokens, emb_model=emb_model, question = example['question'])
        pkes_word_emb, pkes_word_delta, pkes_word_grad  = pke_pipe_across_words(element, emb_model=emb_model, tokenizer=llm.tokenizer, consider_types=True)
        #{'gen_text' : generated_text, 'entropies' : entropies, 'gen_ids' : gen_ids, 'pkes': pkes, 'true_answer' : example['answer']}#['aliases']}
        uqs.append({'question': example['question'], 
                    'gen_text' : element['generated_text'], 
                    'gen_ids' : gen_ids, 
                    'pkes_token_emb': pkes_token_emb, 
                    'pkes_token_sum' : pkes_token_sum, 
                    'pkes_token_word' : pkes_token_word, 
                    'pke_token_deltas': pke_token_deltas, 
                    'pkes_word_emb': pkes_word_emb, 
                    'pkes_word_delta' : pkes_word_delta, 
                    'pkes_word_grad' : pkes_word_grad, 
                    'true_answer' : example['answer'], 
                    'sampled_tokens' : sampled_tokens,
                    'acc' : element['acc']})
    
    
    save(uqs, f'{exp_name}_{ds_name}_uqs.pkl')
        
if __name__ == '__main__':
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    logging.info('STARTING `compute_uncertainty`!')
    main(args)
    logging.info('FINISHED `compute_uncertainty`!')