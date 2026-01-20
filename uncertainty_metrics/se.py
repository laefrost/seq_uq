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

def predictive_entropy_rao(probs, weights = None):
    print("Proooooobs ", probs)
    print("Weeeiiights ", weights)
    if weights is None:
        entropy = - np.sum(probs * np.log(probs))
    else: 
        weights = np.array(weights)
        probs = np.array(probs)
        entropy = - np.sum(weights * probs * np.log(probs))
    return entropy

def compute_se_across_subsequences(cluster_ids_across_steps, seq_tokens, mode = 'complete', weights = None): 
    entropies = []
    counter = 0
    # Compute semantic entropy.
    if weights is not None: 
        for ids, probs, weight in zip(cluster_ids_across_steps, seq_tokens, weights): 
            if mode == 'complete':
                probs_step = probs['alternative_sequence_probs']
            else: 
                probs_step = probs['alternative_token_probs']
            semantic_ids = ids['cluster_ids']
            cluster_weights = weight['cluster_weights']
            probs_per_semantic_id = logsumexp_by_id(semantic_ids, probs=probs_step, agg='sum_normalized')
            pe = predictive_entropy_rao(probs_per_semantic_id, cluster_weights)
            entropies.append(pe)
            counter = counter + 1
    else: 
        for ids, probs in zip(cluster_ids_across_steps, seq_tokens): 
            if mode == 'complete':
                probs_step = probs['alternative_sequence_probs']
            else: 
                probs_step = probs['alternative_token_probs']
            semantic_ids = ids['cluster_ids']
            probs_per_semantic_id = logsumexp_by_id(semantic_ids, probs=probs_step, agg='sum_normalized')
            pe = predictive_entropy_rao(probs_per_semantic_id)
            entropies.append(pe)
            counter = counter + 1
    return entropies


def generate_semantic_subsequence_ids(seq_tokens, question, ellm, mode = 'adapted'): 
    cluster_ids_across_steps = []
    cluster_weights_across_steps = []
    MAX_BATCH = 32
    # for s, step in enumerate(seq_tokens): 
    #     decoded_seqs = step.get('alternative_sequence_decoded', None)
    #     set_step = set(tuple(sublist) for sublist in decoded_seqs)
    #     if len(set_step) == 1: 
    #         cluster_ids = [0] * len(decoded_seqs)
    #     else: 
    #         # decoded_seqs = [tokenizer.decode(ids) for ids in current_step]
            
    #         print('alternative_sequence_decoded ', decoded_seqs)
    #         cluster_ids = get_semantic_ids(strings_list=decoded_seqs, model = ellm, example=question, mode=mode)
    #     cluster_ids_across_steps.append({'cluster_ids' : cluster_ids})
    print('------------------------------------- Sequence Length : ', len(seq_tokens))
    
    for s, step in enumerate(seq_tokens): 
        decoded_seqs = step.get('alternative_sequence_decoded', None) 
        set_step = set(tuple(sublist) for sublist in decoded_seqs)
        print('s----------------------', s, len(decoded_seqs))
        if len(set_step) == 1: 
            cluster_ids = [0] * len(decoded_seqs)
            cluster_weights = [1]
        else:  
            # print(decoded_seqs)   
            # indices = []   
            # batched_pairs = [] 
            # batched_paris_dict = []
            # equiv_pairs = []
            # for i, string1 in enumerate(decoded_seqs):
            #     for j in range(i+1, len(decoded_seqs)):
            #         string2 = decoded_seqs[j]
            #         if string1 == string2 or string1 in string2 or string2 in string1:
            #             continue
            #         if (string1, string2) in batched_pairs or (string2, string1) in batched_pairs:
            #             for entry in batched_paris_dict:
            #                 if entry['s1'] == string1 and entry['s2'] == string2 or entry['s1'] == string2 and entry['s2'] == string1:
            #                     idx_scores = entry['idx_scores']
            #                     break
            #         else: 
            #             indices.append((i,j))
            #             batched_pairs.append((string1, string2))
            #             idx_scores = len(batched_pairs)-1
                        
            #         batched_paris_dict.append({'s1': string1, 's2': string2, 'i': i, 'j': j, 'idx_scores':idx_scores})
                
            #         print(string1)
            #         print(string2)
                    
            # all_scores = []
            # print('Number of unique pairs ', len(batched_pairs))
            # print('Number of different pairs ', len(batched_paris_dict))
            # for b in range(0, len(batched_pairs), MAX_BATCH):
            #     sub = batched_pairs[b:b+MAX_BATCH]
            #     scores = ellm.check_implication_batch(sub, question, mode)
            #     all_scores.extend(scores)
            
            # print('Number of scores ', len(all_scores))
            # score_matrix = np.full((len(decoded_seqs), len(decoded_seqs)), np.nan)

            # for idx, score in enumerate(all_scores):
            #     i = indices[idx][0]
            #     j = indices[idx][1]
            #     score_matrix[i, j] = score
            #     score_matrix[j, i] = score
                
            # for idx, pair in enumerate(batched_paris_dict):
            #     i = pair['i']
            #     j = pair['j']
            #     score_matrix[i, j] = all_scores[pair['idx_scores']]
            #     score_matrix[j, i] = all_scores[pair['idx_scores']]

            # print(score_matrix)
            # entailment_score = 1 if mode == 'data' else 2 
            
            # score_matrix = np.nan_to_num(score_matrix, nan=entailment_score)    
            # print(decoded_seqs)   
            batched_pairs = [] 
            pair_to_idx = {}  # Map (string1, string2) -> index in batched_pairs
            pair_mappings = []  # List of (i, j, score_idx) for matrix population
            
            for i, string1 in enumerate(decoded_seqs):
                for j in range(i+1, len(decoded_seqs)):
                    string2 = decoded_seqs[j]
                    
                    # Skip identical or substring pairs
                    if string1 == string2 or string1 in string2 or string2 in string1:
                        continue
                    
                    # Check if we've already seen this pair
                    pair_key = (string1, string2)
                    reverse_pair_key = (string2, string1)
                    
                    if pair_key in pair_to_idx:
                        score_idx = pair_to_idx[pair_key]
                    elif reverse_pair_key in pair_to_idx:
                        score_idx = pair_to_idx[reverse_pair_key]
                    else:
                        # New unique pair
                        score_idx = len(batched_pairs)
                        batched_pairs.append((string1, string2))
                        pair_to_idx[pair_key] = score_idx
                    
                    pair_mappings.append((i, j, score_idx))
                    #print(string1)
                    #print(string2)
            
            # Get scores for unique pairs only
            all_scores = []
            
            for b in range(0, len(batched_pairs), MAX_BATCH):
                sub = batched_pairs[b:b+MAX_BATCH]
                scores = ellm.check_implication_batch(sub, question, mode)
                all_scores.extend(scores)
            
            # Populate score matrix
            # print('Number of scores:', len(all_scores), scores)
            # print(pair_mappings)
            score_matrix = np.full((len(decoded_seqs), len(decoded_seqs)), np.nan)
            
            for i, j, score_idx in pair_mappings:
                score = all_scores[score_idx]
                score_matrix[i, j] = score
                score_matrix[j, i] = score
            
            entailment_score = 1 if mode == 'data' else 2 
            score_matrix = np.nan_to_num(score_matrix, nan=entailment_score)
            print(score_matrix)       
            
            cluster_ids = [-1] * len(decoded_seqs)
            cluster_weights = []
            next_id = 0
            for i, string1 in enumerate(decoded_seqs):
                # Check if string1 already has an id assigned.
                if cluster_ids[i] == -1:
                    # If string1 has not been assigned an id, assign it next_id.
                    cluster_ids[i] = next_id
                    for j in range(i+1, len(decoded_seqs)):
                        # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                        # if are_equivalent(string1, strings_list[j]):
                        #    semantic_set_ids[j] = next_id
                        # if score_matrix[i, j] == entailment_score:
                        if score_matrix[i, j] in [2]:
                            cluster_ids[j] = next_id
                    next_id += 1

            assert -1 not in cluster_ids
            
            cluster_ids = np.array(cluster_ids)
            unique_cids = np.unique(cluster_ids)
            for c_id in unique_cids: 
                # print('unique id ', c_id)
                # print('cluster_ids before where:', cluster_ids)
                # print('cluster_ids shape before where:', cluster_ids.shape)
                # print('type of c_id:', type(c_id), c_id)
                
                # Check the comparison result
                # comparison = cluster_ids == c_id
                # print('comparison result:', comparison)
                # print('comparison type:', type(comparison))
                # print('comparison shape:', comparison.shape if isinstance(comparison, np.ndarray) else 'not an array')
                if 1 in all_scores:
                    indices = np.where(cluster_ids == c_id)[0]
                    c_weights = []
                    for idx in indices: 
                        row = score_matrix[idx]
                        relevant_entries = row[row == 0]
                        # Neutral to all other sequences --> drop it
                        if len(relevant_entries) == 0: 
                            c_weights.append(0)
                        else: 
                            c_weights.append(1) #len(relevant_entries) / len(row))
                    cluster_weights.append(np.mean(c_weights))
                # there are no neutral tokens
                else: 
                    cluster_weights = [1] * unique_cids
                
                
        cluster_ids_across_steps.append({'cluster_ids' : cluster_ids})
        cluster_weights_across_steps.append({'cluster_weights' : cluster_weights})
        assert len(cluster_ids) == len(decoded_seqs)
    
    return cluster_ids_across_steps, cluster_weights_across_steps
            
        