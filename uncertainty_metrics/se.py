import numpy as np

def get_semantic_ids(strings_list, model, strict_entailment=False, example=None, mode = 'adapted'):
    def are_equivalent(text1, text2):
        #print(text1, text2)
        implication_1, implication_2 = None, None
        while implication_1 not in [0, 1, 2]: 
            implication_1 = model.check_implication(text1, text2, question=example, mode = mode)
        
        while implication_2 not in [0, 1, 2]:
            implication_2 = model.check_implication(text2, text1, question=example, mode = mode)  
        # print('implcations ',implication_1, implication_2)
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

        if strict_entailment:
            semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)

        else:
            implications = [implication_1, implication_2]
            # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
            semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

        #print('semantically equivalent: ', semantically_equivalent)
        
        return semantically_equivalent

    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    next_id = 0
    for i, string1 in enumerate(strings_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i+1, len(strings_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids

    return semantic_set_ids


def logsumexp_by_id(semantic_ids, probs, agg='sum_normalized'):
    """Sum probabilities with the same semantic id.

    Log-Sum-Exp because input and output probabilities in log space.
    """
    unique_ids = sorted(list(set(semantic_ids)))
    # print(unique_ids)
    # print(list(range(len(unique_ids))))
    assert unique_ids == list(range(len(unique_ids)))
    probs_per_semantic_id = []
    sum_clusters_probs = np.sum(probs)
    for uid in unique_ids:
        # Find positions in `semantic_ids` which belong to the active `uid`.
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        # Gather log likelihoods at these indices.
        id_prob = [probs[i] for i in id_indices]
        if agg == 'sum_normalized':
            # log_<
            # lik_norm = id_log_likelihoods - np.prod(log_likelihoods)
            # log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            # logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
            sum_token_prob = np.sum(id_prob)
            
            prob_norm = sum_token_prob / sum_clusters_probs
            
        else:
            raise ValueError
        probs_per_semantic_id.append(prob_norm)

    return probs_per_semantic_id

def predictive_entropy_rao(probs):
    entropy = - np.sum(probs * np.log(probs))
    return entropy

def compute_se_across_subsequences(cluster_ids_across_steps, seq_tokens): 
    entropies = []
    counter = 0
    # Compute semantic entropy.
    for ids, probs in zip(cluster_ids_across_steps, seq_tokens): 
        probs_step = probs['p_s']
        semantic_ids = ids['cluster_ids']
        probs_per_semantic_id = logsumexp_by_id(semantic_ids, probs=probs_step, agg='sum_normalized')
        pe = predictive_entropy_rao(probs_per_semantic_id)
        entropies.append(pe)
        counter = counter + 1
    return entropies


def generate_semantic_subsequence_ids(seq_tokens, question, ellm, mode = 'adapted'): 
    cluster_ids_across_steps = []
    for s, step in enumerate(seq_tokens): 
        current_step = step.get('s', None)
        print(current_step)
        set_step = set(tuple(sublist) for sublist in current_step)
        if len(set_step) == 1: 
            cluster_ids = [0] * len(current_step)
        else: 
            # decoded_seqs = [tokenizer.decode(ids) for ids in current_step]
            decoded_seqs = step.get('s_decoded', None)
            print('decoded_seqs ', decoded_seqs)
            cluster_ids = get_semantic_ids(strings_list=decoded_seqs, model = ellm, example=question, mode=mode)
        cluster_ids_across_steps.append({'cluster_ids' : cluster_ids})
        
    return cluster_ids_across_steps