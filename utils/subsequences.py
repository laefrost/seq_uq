import numpy as np
from copy import deepcopy

def generate_subsequences(sampled_tokens, tokenizer): 
    seq_tokens = []
    current_probs = []
    for i, items in enumerate(sampled_tokens):
        seq_step = list()
        seq_probs = []
        s_str = []
        seq_step_decoded = []
        current_seq = items['current_seq']
        # current_seq_decoded = items['current_seq_decoded']
        current_prob = items['current_prob']
        
        for token in items['sampled_tokens']: 
            tmp_tokens = deepcopy(current_seq)
            tmp_tokens.extend([token['token_id']])
            seq_step.append(tmp_tokens)
            seq_step_decoded.append(tokenizer.decode(current_seq + [token['token_id']], skip_special_tokens=True))#(current_seq_decoded + ' ' + token['token_str'])
            tmp = current_probs + [token['prob']]
            seq_probs.append(np.prod(tmp))
            s_str.append(token['token_str'])
            
        current_probs.append(current_prob)
        seq_tokens.append({'s' : seq_step, 'p_s' : seq_probs, 's_decoded' : seq_step_decoded, 's_str' : s_str})

    return current_probs, seq_tokens


def is_subsequence(small, large):
    """Return True if `small` is a subsequence of `large`."""
    it = iter(large)
    return all(c in it for c in small)

# def is_subsequence(small, large):
#     """Return True if `small` is a subsequence of `large`, by words."""
#     small_words = small.split()
#     large_words = large.split()
#     it = iter(large_words)
#     return all(word in it for word in small_words)

def remove_subsequences(strings):
    """
    Remove strings that are subsequences of a strictly longer string.
    Keep all strings of maximum length.
    """
    strings = sorted(strings, key=len, reverse=True)
    result = []

    for s in strings:
        # Only remove if s is a subsequence of a strictly longer string in result
        if not any(len(t) > len(s) and is_subsequence(s, t) for t in result):
            result.append(s)

    return result[::-1]  # Optional: restore original order


    
    
    