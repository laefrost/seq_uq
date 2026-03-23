import numpy as np
from copy import deepcopy
import torch
import re
import itertools
import random

SEED = 6400
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True) 

def remove_subsequences(sequences, probs, probs_tokens):
    """
    Filter out any sequence that is a strict substring of another sequence in the list.

    Args:
        sequences (list[str]): List of decoded text sequences.
        probs (list[float]): Cumulative probabilities corresponding to each sequence.
        probs_tokens (list[float]): Token-level probabilities corresponding to each sequence.

    Returns:
        tuple: Filtered (sequences, probs, probs_tokens), with subsequences removed.
    """
    keep = [True] * len(sequences)

    for i, seq_i in enumerate(sequences):
        if not keep[i]:
            continue
        for j, seq_j in enumerate(sequences):
            if i != j and keep[j]:
                # if seq_i is entirely inside seq_j, drop seq_i
                if seq_i in seq_j and seq_i != seq_j:
                    keep[i] = False
                    break

    filtered_sequences = [s for s, k in zip(sequences, keep) if k]
    filtered_probs = [p for p, k in zip(probs, keep) if k]
    filtered_token_probs = [p for p, k in zip(probs_tokens, keep) if k]

    return filtered_sequences, filtered_probs, filtered_token_probs

def generate_words(token_ids, tokenizer): 
    """
    Group a flat list of token IDs into words and track which token IDs belong to each word.

    Word boundaries are detected by leading space characters (including unicode variants
    such as ▁ and Ġ) and by punctuation tokens that always start a new segment.

    Args:
        token_ids (list[int]): Ordered list of token IDs from the model output.
        tokenizer: HuggingFace tokenizer used to convert IDs to strings.

    Returns:
        tuple:
            generated_words (list[str]): Decoded surface-form words.
            tokens_ids_across_words (list[list[int]]): For each word, the list of token IDs
                that compose it.
    """
    generated_words = []
    tokens_ids_across_words = []
    tokens_ids_per_word = []
    current_text_tokens = []

    start_new = False

    for token in token_ids:  
        token_text = tokenizer.convert_ids_to_tokens(token, skip_special_tokens = False)
        if token_text.startswith((',', '.', '!', "\"", "?", ";", ":", "(", ")", "`", "´", 
                                  '\u2581(', '</s>', '\u2581\"', '\u2581(', '\u2581\'', "\u2581`", "\u2581´", 
                                  '\u0120(', '<|im_end|>', '\u0120\"', '\u0120(', '\u0120\'', "\u0120`", "\u0120´", 
                                  '<|end|>')): #or token_text in (',', '.', '!', '"', '?', ';', ':', "'", '(', ')', '</s>'):
            if current_text_tokens:
                word = tokenizer.convert_tokens_to_string(current_text_tokens)
                tokens_ids_across_words.append(tokens_ids_per_word)
                generated_words.append(word) 
            start_new = True
            tokens_ids_per_word = [token]
            current_text_tokens = [token_text]
        elif token_text.startswith(' ') or token_text.startswith('\u2581') or token_text.startswith('\u0120') or start_new:
            # Save previous word if it exists
            if current_text_tokens:
                word = tokenizer.convert_tokens_to_string(current_text_tokens)
                tokens_ids_across_words.append(tokens_ids_per_word)
                generated_words.append(word)
            # Start new word
            current_text_tokens = [token_text]
            tokens_ids_per_word = [token]
            start_new = False
        else:
            # Continue building current word
            current_text_tokens.append(token_text)
            tokens_ids_per_word.append(token)
    
    if current_text_tokens:
        word = tokenizer.convert_tokens_to_string(current_text_tokens)
        generated_words.append(word)
        tokens_ids_across_words.append(tokens_ids_per_word)
        
    return generated_words, tokens_ids_across_words

import torch
import torch.nn.functional as F

def top_p_scaling(logits, p=0.9, temperature=1.0):
    """
    Apply temperature scaling followed by top-p (nucleus) filtering to a logits tensor.

    Tokens outside the top-p probability mass are masked to -inf so they receive
    zero probability after softmax. At least one token (the highest-ranked) is
    always retained.

    Args:
        logits (torch.Tensor): 1-D or batched logits tensor of shape (..., vocab_size).
        p (float): Cumulative probability threshold for nucleus filtering. Default 0.9.
        temperature (float): Softmax temperature; values < 1 sharpen, > 1 flatten the
            distribution. Default 1.0.

    Returns:
        torch.Tensor: Filtered logits of the same shape as the input, with out-of-nucleus
            positions set to -inf.
    """
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    filtered_logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)
    
    return filtered_logits
    
def generate_subsequences(step_sequences, tokenizer, gen_ids, sampling_k = 10, scaling_p = None, selection_p = 0.9, method = "sampling", question = ""): 
    """
    For each generation step, sample or select alternative next-token continuations and
    compute their associated probabilities, entropies, and negative log-probabilities.

    Three sampling strategies are supported:
        - "sampling"  : Multinomial sampling over the full (optionally top-p-scaled) distribution.
        - "top_k"     : Deterministic selection of the k highest-probability tokens.
        - anything else: Nucleus (top-p) selection capped at `max_k=20` tokens.

    Args:
        step_sequences (list[dict]): One entry per generation step, each containing:
            - 'current_seq' (list[int]): Token IDs up to and including the current token.
            - 'prev_seq'    (list[int]): Token IDs up to but not including the current token.
            - 'logits'      (torch.Tensor): Raw logits over the vocabulary at this step.
        tokenizer: HuggingFace tokenizer for decoding token IDs to strings.
        gen_ids (list[int]): The actually generated token ID at each step (used to look up
            the current token's probability).
        sampling_k (int): Number of alternative tokens to sample/select per step. Default 10.
        scaling_p (float | None): If set, apply top-p scaling to logits before sampling.
            Default None (no scaling).
        selection_p (float): Nucleus probability threshold used in the nucleus method. Default 0.9.
        method (str): Sampling strategy — "sampling", "top_k", or nucleus (any other string).
            Default "sampling".
        question (str): The prompt/question string prepended when building decoded alternatives.
            Default "".

    Returns:
        list[dict]: One dict per step with keys:
            'prev_seq', 'prev_seq_decoded', 'prev_seq_question_decoded',
            'current_seq', 'current_prob', 'entropy', 'ln_prob',
            'alternative_sequence_tokens', 'alternative_sequence_probs',
            'alternative_token_probs', 'alternative_sequence_decoded',
            'alternative_sequence_question_decoded', 'alternative_tokens_str'.
    """
    seq_tokens = []
    current_probs = []
    for i, items in enumerate(step_sequences):
        seq_step = list()
        seq_probs = []
        token_probs = []
        s_str = []
        seq_step_decoded = []
        
        current_seq = items['current_seq'].copy()
        prev_seq = items['prev_seq'].copy()
        
        logits = items['logits']
        
        if scaling_p is not None:
            # Example usage:
            used_logits = top_p_scaling(logits, p=scaling_p, temperature=1.0)   
        else: 
            used_logits = logits

        probs_og = torch.softmax(used_logits, dim=-1)
        if method == "sampling":   
            sampled_ids = torch.multinomial(probs_og, num_samples=sampling_k, replacement=True)
            probs = probs_og
            current_prob = float(probs[gen_ids[i]].item())
        elif method == "top_k":
            topk_logits, topk_indices = torch.topk(used_logits.squeeze(0), k=sampling_k)
            topk_probs = torch.softmax(topk_logits, dim=-1)
            full_probs = torch.zeros_like(probs_og)
            full_probs.scatter_(0, topk_indices, topk_probs)
            probs = full_probs
            sampled_ids = topk_indices
            if float(probs[gen_ids[i]].item()) == 0: 
                current_prob = probs_og[gen_ids[i]].item()
            else: 
                current_prob = float(probs[gen_ids[i]].item())
        else: 
            sorted_logits, sorted_indices = torch.sort(used_logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            in_nucleus = cumulative_probs < selection_p
            in_nucleus = in_nucleus.squeeze(0)  # (vocab,)
            first_false = (cumulative_probs.squeeze(0) >= selection_p).float().argmax()
            in_nucleus[first_false] = True
            max_k = 20

            rank = torch.arange(sorted_probs.size(-1), device=logits.device)
            rank = rank.view(*([1] * (sorted_probs.ndim - 1)), -1)  # broadcast

            topk_mask = rank < max_k

            nucleus_counts = in_nucleus.sum(dim=-1, keepdim=True)
            final_mask = torch.where(nucleus_counts > max_k, topk_mask, in_nucleus)
            sampled_ids = sorted_indices[final_mask]
            masked_logits = sorted_logits.masked_fill(~final_mask, float('-inf'))
            masked_probs = torch.softmax(masked_logits, dim=-1) 
            scaled_probs = torch.zeros_like(sorted_probs.squeeze())    # (vocab,)
            probs = scaled_probs.scatter(0, sorted_indices.squeeze(), masked_probs.squeeze(0))

            if float(probs[gen_ids[i]].item()) == 0: 
                current_prob = float(probs_og[gen_ids[i]].item())
            else: 
                current_prob = float(probs[gen_ids[i]].item())
        
        for token_id in sampled_ids.tolist():
            token_str = tokenizer.decode([token_id], skip_special_tokens=False)
            token_prob = float(probs[token_id].item())
            if token_prob == 0:
                token_prob = 0.0000001
            tmp_tokens = deepcopy(prev_seq)
            tmp_tokens.extend([token_id])
            seq_step.append(tmp_tokens)
            seq_step_decoded.append(tokenizer.decode(prev_seq + [token_id], skip_special_tokens=False))#(current_seq_decoded + ' ' + token['token_str'])
            tmp = current_probs + [token_prob]
            seq_probs.append(np.prod(tmp))
            s_str.append(token_str)
            token_probs.append(token_prob)
        
        ln_prob = -np.log(current_prob)
        log_vals = torch.where(
            probs == 0,
            torch.zeros_like(probs),
            torch.log(probs)
        )
        entropy = - torch.sum(probs * log_vals)
        
        current_probs.append(current_prob)
        seq_tokens.append({'prev_seq': prev_seq,
                           'prev_seq_decoded': tokenizer.decode(prev_seq, skip_special_tokens=True), 
                           'prev_seq_question_decoded': question + ' ' + tokenizer.decode(prev_seq, skip_special_tokens=True), 
                           'current_seq': current_seq,
                           'current_prob' : current_prob, 
                           'entropy' : entropy, 
                           'ln_prob' : ln_prob,
                           'alternative_sequence_tokens' : seq_step, 
                           'alternative_sequence_probs' : seq_probs, 
                           'alternative_token_probs' : token_probs,
                           'alternative_sequence_decoded' : seq_step_decoded,
                           'alternative_sequence_question_decoded': [question + ' ' + s for s in seq_step_decoded],
                           'alternative_tokens_str' : s_str})

    return seq_tokens


def generate_word_subsequences(seq_tokens, generated_words, word_ids, question, generated_text, tokenizer):
    """
    Aggregate token-level subsequence data into word-level subsequence data.

    For each word in `generated_words`, the corresponding token entries in `seq_tokens`
    are combined: probabilities are multiplied across tokens, alternative continuations
    are merged, and strict sub-continuations are pruned via `remove_subsequences`.

    Args:
        seq_tokens (list[dict]): Token-level data as returned by `generate_subsequences`.
        generated_words (list[str]): Surface-form words in generation order, as returned
            by `generate_words`.
        word_ids (list[list[int]]): For each word, the list of token IDs that compose it,
            as returned by `generate_words`.
        question (str): The prompt string prepended to all decoded sequences.
        generated_text (str): The full decoded generated text (used for sequence building).
        tokenizer: HuggingFace tokenizer for decoding token ID sequences.

    Returns:
        list[dict]: One dict per word with keys:
            'prev_seq_decoded', 'prev_seq_question_decoded',
            'current_seq', 'current_prob', 'entropy', 'ln_prob',
            'alternative_sequence_probs', 'alternative_token_probs',
            'alternative_sequence_question_decoded', 'current_word'.
    """
    seq_words = []
    seq_idx = 0
    text_and_question = question + ' ' + generated_text
    prev_seq = ""
    for w, word in enumerate(generated_words):
        ids = word_ids[w]
        if len(ids) == 1:
            seq_data = seq_tokens[seq_idx]
            end_idx = seq_idx + 1
            alternative_sequences = [
                question + ' ' + t for t in seq_data['alternative_sequence_decoded']
            ]
            current_prob = seq_data['current_prob']
            entropy = seq_data['entropy']
            ln_prob = seq_data['ln_prob']
            alternative_probs = seq_data['alternative_sequence_probs']
            alternative_token_probs = seq_data['alternative_token_probs']
        else: 
            alternative_sequences = []
            alternative_probs = []
            alternative_token_probs = []
            current_probs = []
            entropies = []
            end_idx = seq_idx + len(ids)
            prev_probs_words = [1] * len(seq_tokens[seq_idx]['alternative_sequence_decoded'])
            prev_prob = 1            

            for i in list(range(seq_idx, end_idx)):
                seq_data = seq_tokens[i]
                n  = len(seq_tokens[i]['alternative_sequence_decoded'])
                entropies.append(seq_data['entropy'])
                prev_probs = [prev_prob] * n 
                alternative_sequences.extend([
                    question + ' ' + t for t in seq_data['alternative_sequence_decoded']
                ])
            
                alternative_probs.extend([x * y for x, y in zip(prev_probs, seq_data['alternative_sequence_probs'])])
                alternative_token_probs.extend([x * y for x, y in zip(prev_probs, seq_data['alternative_token_probs'])])
                current_probs.append(seq_data['current_prob'])
                prev_prob = np.prod(current_probs)

            alternative_sequences, alternative_probs, alternative_token_probs = remove_subsequences(
                alternative_sequences, alternative_probs, alternative_token_probs
            )
             
            current_prob = np.prod(current_probs) 
            entropy = np.sum(entropies)   
            ln_prob = - np.log(current_prob)        
                
        current_sequence_token_ids_unpacked = list(itertools.chain.from_iterable(word_ids[:w+1]))
        current_sequence = tokenizer.decode(current_sequence_token_ids_unpacked, skip_special_tokens = False)
        seq_idx = end_idx
        
        seq_words.append({
            'prev_seq_decoded': prev_seq,
            'prev_seq_question_decoded' : question + ' ' + prev_seq,
            'current_seq': current_sequence,
            'current_prob': current_prob,
            'entropy' : entropy, 
            'ln_prob' : ln_prob,
            'alternative_sequence_probs': alternative_probs,
            'alternative_token_probs' : alternative_token_probs, 
            'alternative_sequence_question_decoded': alternative_sequences,
            'current_word': word, 
        })
        
        prev_seq = current_sequence
        
    if len(seq_words) != len(generated_words):
        print(f"WARNING: Length mismatch! seq_words={len(seq_words)}, generated_words={len(generated_words)}")
        print(f"Difference: {len(generated_words) - len(seq_words)}")

    return seq_words