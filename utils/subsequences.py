import numpy as np
from copy import deepcopy
import torch
import re
import itertools


def remove_subsequences(sequences, probs, probs_tokens):
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
    generated_words = []
    tokens_ids_across_words = []
    tokens_ids_per_word = []
    current_text_tokens = []

    start_new = False

    for token in token_ids:  
        token_text = tokenizer.convert_ids_to_tokens(token, skip_special_tokens = False)
        if token_text.startswith((',', '.', '!', "\"", "?", ";", ":", "(", ")", "`", "´", '\u2581(', '</s>', 
                                  '\u2581\"', '\u2581(', '\u2581\'', "\u2581`", "\u2581´" )): #or token_text in (',', '.', '!', '"', '?', ';', ':', "'", '(', ')', '</s>'):
            if current_text_tokens:
                word = tokenizer.convert_tokens_to_string(current_text_tokens)
                tokens_ids_across_words.append(tokens_ids_per_word)
                generated_words.append(word) 
            start_new = True
            tokens_ids_per_word = [token]
            current_text_tokens = [token_text]
        elif token_text.startswith(' ') or token_text.startswith('\u2581') or start_new:
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
    # Apply temperature scaling
    logits = logits / temperature
    
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # Compute softmax probabilities for sorted logits
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    # Keep at least one token (the first one)
    sorted_indices_to_remove = cumulative_probs > p
    # Shift right to keep the first token that exceeds threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Set removed indices to -inf in sorted logits
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    
    # Scatter back to original order
    filtered_logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)
    
    return filtered_logits
    
def generate_subsequences(step_sequences, tokenizer, gen_ids, sampling_k = 10, scaling_p = None, selection_p = 0.9, method = "sampling", question = ""): 
    seq_tokens = []
    current_probs = []
    for i, items in enumerate(step_sequences):
        seq_step = list()
        seq_probs = []
        token_probs = []
        s_str = []
        seq_step_decoded = []
        
        current_seq = items['current_seq']
        prev_seq = items['prev_seq']
        
        logits = items['logits']
        
        if scaling_p is not None:
            # Example usage:
            used_logits = top_p_scaling(logits, p=scaling_p, temperature=1.0)   
        else: 
            used_logits = logits

        probs = torch.softmax(used_logits, dim=-1) 
        
        if method == "sampling":   
            sampled_ids = torch.multinomial(probs, num_samples=sampling_k, replacement=True)
        else: 
            # sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            # sorted_probs = F.softmax(sorted_logits, dim=-1)
            # cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # in_nucleus = cumulative_probs <= selection_p
            # in_nucleus[..., 0] = True            
            # sampled_ids = sorted_indices[in_nucleus]
            # print("sampled ids", sampled_ids)
            sorted_logits, sorted_indices = torch.sort(used_logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            in_nucleus = cumulative_probs <= selection_p
            in_nucleus[..., 0] = True  # always keep best token

            max_k = 20

            # rank positions along vocab dim
            rank = torch.arange(sorted_probs.size(-1), device=logits.device)
            rank = rank.view(*([1] * (sorted_probs.ndim - 1)), -1)  # broadcast

            topk_mask = rank < max_k

            nucleus_counts = in_nucleus.sum(dim=-1, keepdim=True)
            final_mask = torch.where(nucleus_counts > max_k, topk_mask, in_nucleus)

            # deterministic selected token ids (variable count per row)
            sampled_ids = sorted_indices[final_mask]
        
        
            print(len(sampled_ids))
        current_prob = float(probs[gen_ids[i]].item())
        
        for token_id in sampled_ids.tolist():
            token_str = tokenizer.decode([token_id], skip_special_tokens=False)
            token_prob = float(probs[token_id].item())
            tmp_tokens = deepcopy(prev_seq)
            tmp_tokens.extend([token_id])
            seq_step.append(tmp_tokens)
            seq_step_decoded.append(tokenizer.decode(prev_seq + [token_id], skip_special_tokens=False))#(current_seq_decoded + ' ' + token['token_str'])
            tmp = current_probs + [token_prob]
            seq_probs.append(np.prod(tmp))
            s_str.append(token_str)
            token_probs.append(token_prob)
        
        # print("tokenprobs ", token_probs)
        
        ln_prob = -np.log(current_prob)
        entropy = torch.sum(used_logits * torch.log(used_logits))
        
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
                           'alternative_sequence_question_decoded': question + ' ' + seq_step_decoded[0],
                           'alternative_tokens_str' : s_str})

    return seq_tokens


def generate_word_subsequences(seq_tokens, generated_words, word_ids, question, generated_text, tokenizer):
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
            
            # das hier checken
            for i in list(range(seq_idx, end_idx)):
                seq_data = seq_tokens[i]
                n  = len(seq_tokens[i]['alternative_sequence_decoded'])
                entropies.append(seq_data['entropy'])
                prev_probs = [prev_prob] * n 
                alternative_sequences.extend([
                    question + ' ' + t for t in seq_data['alternative_sequence_decoded']
                ])
                # print("n", n, " len prev probs", len(prev_probs), "len token probs ", len(seq_data['alternative_sequence_probs']))
                #alternative_probs.extend(prev_probs * seq_data['alternative_sequence_probs'])
                #alternative_token_probs.extend(prev_probs * seq_data['alternative_token_probs'])
                alternative_probs.extend([x * y for x, y in zip(prev_probs, seq_data['alternative_sequence_probs'])])
                alternative_token_probs.extend([x * y for x, y in zip(prev_probs, seq_data['alternative_token_probs'])])
                current_probs.append(seq_data['current_prob'])
                prev_prob = seq_data['current_prob']
            
                # Remove duplicate subsequences
            #alternative_sequences, alternative_probs, alternative_token_probs = remove_subsequences(
            #    alternative_sequences, alternative_probs, alternative_token_probs
            #)
            # print("final len", len(alternative_token_probs))    
            current_prob = np.prod(current_probs) 
            entropy = np.sum(entropies)   
            ln_prob = - np.log(current_prob)        
            # for i, seq in enumerate(alternative_sequences):
            #     if seq in text_and_question:
            #         print("In text", seq, generated_text)
            #         alternative_token_probs[i] = current_prob
        
        # TODO: Das vlt checken            
        current_sequence_token_ids_unpacked = list(itertools.chain.from_iterable(word_ids[:w+1]))
        current_sequence = tokenizer.decode(current_sequence_token_ids_unpacked, skip_special_tokens = False)
        seq_idx = end_idx
        
        # Append word-level data
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
    
    # print(len(seq_words))
    return seq_words



# def generate_subsequences(sampled_tokens, tokenizer): 
#     seq_tokens = []
#     current_probs = []
#     for i, items in enumerate(sampled_tokens):
#         seq_step = list()
#         seq_probs = []
#         token_probs = []
#         s_str = []
#         seq_step_decoded = []
#         current_seq = items['current_seq']
#         prev_seq = items['prev_seq']
#         # current_seq_decoded = items['current_seq_decoded']
#         current_prob = items['current_prob']
        
#         for token in items['sampled_tokens']: 
#             tmp_tokens = deepcopy(prev_seq)
#             tmp_tokens.extend([token['token_id']])
#             seq_step.append(tmp_tokens)
#             seq_step_decoded.append(tokenizer.decode(prev_seq + [token['token_id']], skip_special_tokens=True))#(current_seq_decoded + ' ' + token['token_str'])
#             tmp = current_probs + [token['prob']]
#             seq_probs.append(np.prod(tmp))
#             s_str.append(token['token_str'])
#             token_probs.append(token['prob'])
            
#         current_probs.append(current_prob)
#         seq_tokens.append({'prev_seq': prev_seq,
#                            'prev_seq_decoded': tokenizer.decode(prev_seq, skip_special_tokens=True), 
#                            'current_seq': current_seq,
#                            'current_prob' : current_prob, 
#                            'alternative_sequence_tokens' : seq_step, 
#                            'alternative_sequence_probs' : seq_probs, 
#                            'alternative_token_probs' : token_probs,
#                            'alternative_sequence_decoded' : seq_step_decoded,
#                            'alternative_tokens_str' : s_str})

#     return current_probs, seq_tokens

# def generate_word_subsequences(seq_tokens, generated_text, question, token_ids, tokenizer):
#     """
#     Generate word-level subsequences from token-level data.
    
#     Args:
#         seq_tokens: List of token-level sequence data
#         generated_text: The full generated text string
#         question: The input question/prompt
#         tokens: List of token IDs
#         tokenizer: Tokenizer object for decoding
    
#     Returns:
#         List of word-level sequence dictionaries
#     """
#     seq_words = []
#     generated_words = []
#     tokens_ids_across_words = []
#     tokens_ids_per_word = []
#     token_probs_per_word = []
#     current_text_tokens = []

#     start_new = False

#     for token in token_ids:  
#         token_text = tokenizer.convert_ids_to_tokens(token, skip_special_tokens = False)
#         if token_text.startswith((',', '.', '!', "\"", "?", ";", ":", "(", ")", "`", "´", '\u2581(', '</s>', 
#                                   '\u2581\"', '\u2581(', '\u2581\'', "\u2581`", "\u2581´" )): #or token_text in (',', '.', '!', '"', '?', ';', ':', "'", '(', ')', '</s>'):
#             if current_text_tokens:
#                 word = tokenizer.convert_tokens_to_string(current_text_tokens)
#                 tokens_ids_across_words.append(tokens_ids_per_word)
#                 generated_words.append(word) 
#             start_new = True
#             tokens_ids_per_word = [token]
#             current_text_tokens = [token_text]
#         elif token_text.startswith(' ') or token_text.startswith('\u2581') or start_new:
#             # Save previous word if it exists
#             if current_text_tokens:
#                 word = tokenizer.convert_tokens_to_string(current_text_tokens)
#                 tokens_ids_across_words.append(tokens_ids_per_word)
#                 generated_words.append(word)
#             # Start new word
#             current_text_tokens = [token_text]
#             tokens_ids_per_word = [token]
#             start_new = False
#         else:
#             # Continue building current word
#             current_text_tokens.append(token_text)
#             tokens_ids_per_word.append(token)
    
#     if current_text_tokens:
#         word = tokenizer.convert_tokens_to_string(current_text_tokens)
#         generated_words.append(word)
#         tokens_ids_across_words.append(tokens_ids_per_word)
        
    
#     token_idx = 0
#     word_idx = 0
#     prev_seq = question
#     current_sequence_token_ids = []
    
#     while word_idx < len(generated_words):
#         torch.cuda.empty_cache()
        
#         # Check if we've run out of tokens
#         if token_idx >= len(token_ids):
#             print(f"Warning: Ran out of tokens at word_idx={word_idx}/{len(generated_words)}")
#             break
        
#         current_word = generated_words[word_idx]
#         word_tokens = [token_ids[token_idx]]
#         decoded_tokens = tokenizer.decode(word_tokens)
        
#         # Track how many additional words we need to merge
#         words_to_merge = 0
        
#         # Case 1: Single token doesn't fully represent the word
#         # Need to accumulate more tokens
#         if len(decoded_tokens) < len(current_word):
#             while decoded_tokens != current_word and token_idx + 1 < len(token_ids):
#                 token_idx += 1
#                 word_tokens.append(token_ids[token_idx])
#                 decoded_tokens = tokenizer.decode(word_tokens, skip_special_tokens = False)
                
#                 # Safety check
#                 if len(word_tokens) > 20:
#                     print(f"Warning: Accumulated {len(word_tokens)} tokens for word '{current_word}'")
#                     break
        
#         # Case 2: Single token represents multiple words
#         # Need to merge words until they match the decoded token(s)
#         else:
#             merged_word = current_word
#             while decoded_tokens != merged_word and word_idx + words_to_merge + 1 < len(generated_words):
#                 words_to_merge += 1
#                 merged_word += generated_words[word_idx + words_to_merge]
                
#                 # Safety check
#                 if words_to_merge > 10:
#                     print(f"Warning: Merged {words_to_merge} words for token '{decoded_tokens}'")
#                     break
            
#             current_word = merged_word
        
#         # Move to next token for next iteration
#         token_idx += 1
#         num_tokens_used = len(word_tokens)
#         start_seq_idx = token_idx - num_tokens_used
#         end_seq_idx = token_idx
        
#         # Aggregate data from multiple tokens if needed
#         if num_tokens_used > 1:
#             alternative_sequences = []
#             alternative_probs = []
#             alternative_token_probs = []
#             current_probs = []
            
#             for idx in range(start_seq_idx, end_seq_idx):
#                 if idx >= len(seq_tokens):
#                     print(f"Warning: seq_token index {idx} out of bounds")
#                     break
                    
#                 seq_data = seq_tokens[idx]
#                 alternative_sequences.extend([
#                     question + ' ' + t for t in seq_data['alternative_sequence_decoded']
#                 ])
#                 alternative_probs.extend(seq_data['alternative_sequence_probs'])
#                 alternative_token_probs.extend(seq_data['alternative_token_probs'])
#                 current_probs.append(seq_data['current_prob'])
            
#             # Remove duplicate subsequences
#             alternative_sequences, alternative_probs, alternative_token_probs = remove_subsequences(
#                 alternative_sequences, alternative_probs, alternative_token_probs
#             )
            
#             # Calculate combined probability
#             current_prob = np.prod(current_probs)
            
#             for i, seq in enumerate(alternative_sequences):
#                 if seq in generated_text:
#                     print("In text", seq, generated_text)
#                     alternative_token_probs[i] = current_probs                     
#         else:
#             # Single token case
#             if start_seq_idx >= len(seq_tokens):
#                 print(f"Warning: seq_token index {start_seq_idx} out of bounds")
#                 break
                
#             seq_data = seq_tokens[start_seq_idx]
#             alternative_sequences = [
#                 question + ' ' + t for t in seq_data['alternative_sequence_decoded']
#             ]
#             current_prob = seq_data['current_prob']
#             alternative_probs = seq_data['alternative_sequence_probs']
#             alternative_token_probs = seq_data['alternative_token_probs']
        
#         # Build current sequence
        
#         current_sequence_token_ids.append(word_tokens)
#         current_sequence_token_ids_unpacked = list(itertools.chain.from_iterable(current_sequence_token_ids))
#         current_sequence = tokenizer.decode(current_sequence_token_ids_unpacked, skip_special_tokens = False)
        
#         # Append word-level data
#         seq_words.append({
#             'prev_seq': prev_seq,
#             'current_seq': current_sequence,
#             'current_prob': current_prob,
#             'alternative_sequence_probs': alternative_probs,
#             'alternative_token_probs' : alternative_token_probs, 
#             'alternative_sequence_decoded': alternative_sequences,
#             'current_word': current_word, 
#             'current_sequence_token_ids' : current_sequence_token_ids
#         })
        
#         # Update for next iteration
#         prev_seq = current_sequence
#         word_idx += words_to_merge + 1
        
#         # Debug output
#         # print(f"Processed {word_idx}/{len(generated_words)} words, {len(seq_words)} seq_words created")
    
#     #print(f"Final: {len(seq_words)} seq_words from {len(generated_words)} generated_words")
    
#     # Validation
#     if len(seq_words) != len(generated_words):
#         print(f"WARNING: Length mismatch! seq_words={len(seq_words)}, generated_words={len(generated_words)}")
#         print(f"Difference: {len(generated_words) - len(seq_words)}")
    
#     return seq_words, generated_words, current_sequence_token_ids



# def generate_word_subsequences(seq_tokens, generated_text, question, tokens, tokenizer): 
#     seq_words = []
#     # for i, instance in enumerate(seq_tokens): 
#     skipped = 0 
#     pattern = r"\(|\)|[0-9]+(?:[.,-][0-9]+)*|[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)*|[.,;?!:]|\n|</s>|'|\"|`|´|-"
#     #r"\([0-9]+(?:[-–][0-9]+)*\)|[0-9]+(?:[.,-][0-9]+)*|[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)*|[.,;?!:]|\n|\(|\)|</s>|\'|\"|\?|\!|\`|\´|\-|-"
    
#     generated_words = re.findall(pattern, generated_text)
#     token_idx = 0
#     skipped_words = 0
    
#     prev_seq = question
#     #print('generated Text', generated_text)
#     #print('Generated Words: ', generated_words)
#     #print(tokens)

#     for w, word in enumerate(generated_words): 
#         torch.cuda.empty_cache()
#         if skipped_words > 0: 
#             skipped_words = skipped_words - 1
#             skipped = skipped - 1 
#             continue
        
#         word_tokens = [tokens[token_idx]]
#         decoded_tokens = tokenizer.decode(word_tokens)        

#         if len(decoded_tokens) < len(word):  
#             #print('lenss ')
#             #print(len(decoded_tokens), len(word)) 
#             while decoded_tokens != word:
#                 token_idx = token_idx + 1 
#                 word_tokens.append(tokens[token_idx])
#                 decoded_tokens = tokenizer.decode(word_tokens)
                
#         else:
#             skipped_words = 0
#             while decoded_tokens != word and skipped_words < len(generated_words):
#                 skipped_words = skipped_words + 1 
#                 word = word + generated_words[w+1]
        
#         token_idx = token_idx + 1 
#         if len(word_tokens) > 1: 
#             alternative_sequences = []
#             alternative_probs = []
#             current_probs = []
#             for i, token in enumerate(word_tokens):
#                 alternative_sequences.extend([question + ' ' + t for t in seq_tokens[w + skipped + i]['alternative_sequence_decoded']])
#                 alternative_probs.extend(seq_tokens[w + skipped + i]['alternative_sequence_probs'])
#                 current_probs.append(seq_tokens[w + skipped + i]['current_prob'])
                
#             alternative_sequences, alternative_probs = remove_subsequences(alternative_sequences, alternative_probs)
#             current_prob = np.prod(current_probs)
            
#             skipped = skipped + len(word_tokens) - 1
#         else: 
#             alternative_sequences = [question + ' ' + t for t in seq_tokens[w + skipped]['alternative_sequence_decoded']]
#             current_prob = seq_tokens[w + skipped]['current_prob']
#             alternative_probs = seq_tokens[w + skipped]['alternative_sequence_probs']
#         # print('Alternative Sequences: ', alternative_sequences)
#         current_sequence = prev_seq + decoded_tokens 
#         seq_words.append({'prev_seq': prev_seq, 
#                             'current_seq': current_sequence, 
#                             'current_prob' : current_prob, 
#                             'alternative_sequence_probs' : alternative_probs, 
#                             'alternative_sequence_decoded' : alternative_sequences, 
#                             'alternative_word_str' : word})
        
#         prev_seq = current_sequence
#     return seq_words


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


    
    
    