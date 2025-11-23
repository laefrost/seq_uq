import numpy as np
from copy import deepcopy
import torch
import re

def remove_subsequences(sequences, probs):
    keep = [True] * len(sequences)

    for i, seq_i in enumerate(sequences):
        if not keep[i]:
            continue
        for j, seq_j in enumerate(sequences):
            if i != j and keep[j]:
                # if seq_i is entirely inside seq_j, drop seq_i
                if seq_i in seq_j:
                    keep[i] = False
                    break

    filtered_sequences = [s for s, k in zip(sequences, keep) if k]
    filtered_probs = [p for p, k in zip(probs, keep) if k]

    return filtered_sequences, filtered_probs



def generate_subsequences(sampled_tokens, tokenizer): 
    seq_tokens = []
    current_probs = []
    for i, items in enumerate(sampled_tokens):
        seq_step = list()
        seq_probs = []
        s_str = []
        seq_step_decoded = []
        current_seq = items['current_seq']
        prev_seq = items['prev_seq']
        # current_seq_decoded = items['current_seq_decoded']
        current_prob = items['current_prob']
        
        for token in items['sampled_tokens']: 
            tmp_tokens = deepcopy(prev_seq)
            tmp_tokens.extend([token['token_id']])
            seq_step.append(tmp_tokens)
            seq_step_decoded.append(tokenizer.decode(prev_seq + [token['token_id']], skip_special_tokens=True))#(current_seq_decoded + ' ' + token['token_str'])
            tmp = current_probs + [token['prob']]
            seq_probs.append(np.prod(tmp))
            s_str.append(token['token_str'])
            
        current_probs.append(current_prob)
        seq_tokens.append({'prev_seq': prev_seq,
                           'current_seq': current_seq,
                           'current_prob' : current_prob, 
                           'alternative_sequence_tokens' : seq_step, 
                           'alternative_sequence_probs' : seq_probs, 
                           'alternative_sequence_decoded' : seq_step_decoded,
                           'alternative_tokens_str' : s_str})

    return current_probs, seq_tokens

def generate_word_subsequences(seq_tokens, generated_text, question, tokens, tokenizer): 
    seq_words = []
    # for i, instance in enumerate(seq_tokens): 
    skipped = 0 
    pattern = r"\(|\)|[0-9]+(?:[.,-][0-9]+)*|[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)*|[.,;?!:]|\n|</s>|'|\"|`|´|-"
    #r"\([0-9]+(?:[-–][0-9]+)*\)|[0-9]+(?:[.,-][0-9]+)*|[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)*|[.,;?!:]|\n|\(|\)|</s>|\'|\"|\?|\!|\`|\´|\-|-"
    
    generated_words = re.findall(pattern, generated_text)
    token_idx = 0
    skipped_words = 0
    
    prev_seq = question
    #print('generated Text', generated_text)
    #print('Generated Words: ', generated_words)
    #print(tokens)

    for w, word in enumerate(generated_words): 
        #print('wooooooooooooooooort ', word)
        torch.cuda.empty_cache()
        if skipped_words > 0: 
            skipped_words = skipped_words - 1
            skipped = skipped - 1 
            continue
        
        #print('Token idx ', token_idx)
        word_tokens = [tokens[token_idx]]
        decoded_tokens = tokenizer.decode(word_tokens)
        #print('workd tokens', word_tokens)
        

        if len(decoded_tokens) < len(word):  
            #print('lenss ')
            #print(len(decoded_tokens), len(word)) 
            while decoded_tokens != word:
                token_idx = token_idx + 1 
                word_tokens.append(tokens[token_idx])
                decoded_tokens = tokenizer.decode(word_tokens)
                #print("decoded tokens ", decoded_tokens)
                #print('workd tokens', word_tokens)
                
        else:
            skipped_words = 0
            while decoded_tokens != word and skipped_words < len(generated_words):
                skipped_words = skipped_words + 1 
                word = word + generated_words[w+1]
        
        token_idx = token_idx + 1 
        if len(word_tokens) > 1: 
            alternative_sequences = []
            alternative_probs = []
            current_probs = []
            for i, token in enumerate(word_tokens):
                alternative_sequences.extend([question + ' ' + t for t in seq_tokens[w + skipped + i]['alternative_sequence_decoded']])
                alternative_probs.extend(seq_tokens[w + skipped + i]['alternative_sequence_probs'])
                current_probs.append(seq_tokens[w + skipped + i]['current_prob'])
                
            alternative_sequences, alternative_probs = remove_subsequences(alternative_sequences, alternative_probs)
            current_prob = np.prod(current_probs)
            
            skipped = skipped + len(word_tokens) - 1
        else: 
            alternative_sequences = [question + ' ' + t for t in seq_tokens[w + skipped]['alternative_sequence_decoded']]
            current_prob = seq_tokens[w + skipped]['current_prob']
            alternative_probs = seq_tokens[w + skipped]['alternative_sequence_probs']
        print('Alternative Sequences: ', alternative_sequences)
        current_sequence = prev_seq + decoded_tokens 
        seq_words.append({'prev_seq': prev_seq, 
                            'current_seq': current_sequence, 
                            'current_prob' : current_prob, 
                            'alternative_sequence_probs' : alternative_probs, 
                            'alternative_sequence_decoded' : alternative_sequences, 
                            'alternative_word_str' : word})
        
        prev_seq = current_sequence
    return seq_words


# def is_subsequence(small, large):
#     """Return True if `small` is a subsequence of `large`."""
#     it = iter(large)
#     return all(c in it for c in small)

# # def is_subsequence(small, large):
# #     """Return True if `small` is a subsequence of `large`, by words."""
# #     small_words = small.split()
# #     large_words = large.split()
# #     it = iter(large_words)
# #     return all(word in it for word in small_words)

# def remove_subsequences(strings):
#     """
#     Remove strings that are subsequences of a strictly longer string.
#     Keep all strings of maximum length.
#     """
#     strings = sorted(strings, key=len, reverse=True)
#     result = []

#     for s in strings:
#         # Only remove if s is a subsequence of a strictly longer string in result
#         if not any(len(t) > len(s) and is_subsequence(s, t) for t in result):
#             result.append(s)

#     return result[::-1]  # Optional: restore original order


    
    
    