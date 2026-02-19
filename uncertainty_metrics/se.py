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

def predictive_entropy_per_topic(semantic_ids, topic_ids, probs): 
    topic_ids = np.asarray(topic_ids)
    semantic_ids = np.asarray(semantic_ids)
    probs = np.asarray(probs)
    unique_topic_ids = set(topic_ids)
    entropies_per_topic = []
    #print("Unique topic ids: ", unique_topic_ids, topic_ids)
    #print("Semantic topic ids: ", semantic_ids)
    #print("probs:", probs)
    for topic_id in unique_topic_ids: 
        # get the entries from the clusters
        relevant_entries = np.where(topic_ids == topic_id)[0]
        topic_cluster_ids = semantic_ids[relevant_entries]
        unique_topic_cluster_ids = set(topic_cluster_ids)
        topic_probs = probs[relevant_entries]
        sum_topic_probs = np.sum(topic_probs)
        probs_per_semantic_id = []
        for uid in unique_topic_cluster_ids:
            id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
            id_prob = [probs[i] for i in id_indices]
            sum_token_prob = np.sum(id_prob)
            # basically the cluster probability 
            prob_norm = sum_token_prob / sum_topic_probs
            probs_per_semantic_id.append(prob_norm)

        #print("normalized probs ", prob_norm)
        #print("probs across clusters in topic: ", probs_per_semantic_id)
        # assert np.sum(probs_per_semantic_id) == 1
        
        cond_se = predictive_entropy_rao(probs_per_semantic_id)
        assert cond_se >= 0
        entropies_per_topic.append(cond_se)
        
    return entropies_per_topic
            

def logsumexp_by_id(semantic_ids, probs, agg='sum_normalized'):
    """Sum probabilities with the same semantic id.

    Log-Sum-Exp because input and output probabilities in log space.
    """
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    probs_per_semantic_id = []
    sum_clusters_probs = np.sum(probs)
    # print("Probs in logsum", probs)
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

def predictive_cond_entropy(topics, pred_entropies_per_topic):
    unique_topic_ids, counts = np.unique(topics, return_counts=True)
    freqs = counts / counts.sum()   # P(topic)
    entropy_vec = np.array(pred_entropies_per_topic, dtype=float)
    cond_entropy = np.sum(freqs * entropy_vec)
    return cond_entropy

def compute_se_across_subsequences(cluster_ids_across_steps, seq_tokens, probs, mode = 'complete', topics = None): 
    entropies = []
    counter = 0
    # Compute semantic entropy.
    if topics is not None: 
        for ids, probs_step, topic in zip(cluster_ids_across_steps, probs, topics): 
            # if mode == 'complete':
            #     probs_step = probs['alternative_sequence_probs']
            # else: 
            #     probs_step = probs['alternative_token_probs']
            semantic_ids = ids['cluster_ids']
            topic_ids = topic['topic_ids']
            pe_topics = predictive_entropy_per_topic(semantic_ids, topic_ids, probs_step)
            #print('pe per topic: ', pe_topics)
            cond_pe = predictive_cond_entropy(topic_ids, pe_topics)
            entropies.append(cond_pe)
    else: 
        for ids, probs_step in zip(cluster_ids_across_steps, probs): 
            # if mode == 'complete':
            #     probs_step = probs['alternative_sequence_probs']
            # else: 
            #     probs_step = probs['alternative_token_probs']
            semantic_ids = ids['cluster_ids']
            probs_per_semantic_id = logsumexp_by_id(semantic_ids, probs=probs_step, agg='sum_normalized')
            pe = predictive_entropy_rao(probs_per_semantic_id)
            entropies.append(pe)
            counter = counter + 1
    return entropies


def generate_semantic_subsequence_ids(seq_tokens, question, ellm, mode = 'adapted'): 
    cluster_ids_across_steps = []
    # cluster_weights_across_steps = []
    topic_ids_across_steps = []
    probs_across_steps = []
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
    
    
    #print('------------------------------------- Sequence Length : ', len(seq_tokens))
    
    for s, step in enumerate(seq_tokens): 
        decoded_seqs = step.get('alternative_sequence_question_decoded', None)
        probs = step.get('alternative_token_probs', None)
        
        # print(decoded_seqs, probs)
        
        unique_elements = list(set(zip(decoded_seqs, probs)))
        # unique_seqs = set(decoded_seqs)
        #print(unique_elements)
        set_step = set(tuple(sublist) for sublist in decoded_seqs)
        #print('s----------------------', s, len(decoded_seqs))
        # if len(set_step) == 1 or len(decoded_seqs) == 1: 
        if len(unique_elements) == 1:
            #cluster_ids = [0] * len(decoded_seqs)
            #topic_ids = [0] * len(decoded_seqs)
            cluster_ids = [0]
            topic_ids = [0]
            probs = [unique_elements[0][1]]
            # cluster_weights = [1]
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
            
            decoded_seqs = [element[0] for element in unique_elements]
            probs = [element[1] for element in unique_elements]
            score_matrix = np.full((len(decoded_seqs), len(decoded_seqs)), np.nan)
            entailment_score = 1 if mode == 'data' else 2 
            contradiction_score = 0
            neutral_score = 1
            np.fill_diagonal(score_matrix, entailment_score)

            
            for i, string1 in enumerate(decoded_seqs):
                for j in range(i+1, len(decoded_seqs)):
                    string2 = decoded_seqs[j]
                    if string1 == string2:
                        score_matrix[i, j], score_matrix[j, i] = entailment_score, entailment_score
                        continue
                    elif string1 in string2 or string2 in string1:
                        score_matrix[i, j], score_matrix[j, i] = neutral_score, neutral_score
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
            
            # Get scores for unique pairs only
            all_scores = []
            
            for b in range(0, len(batched_pairs), MAX_BATCH):
                sub = batched_pairs[b:b+MAX_BATCH]
                scores, contr_prob = ellm.check_implication_batch(sub, question, mode)
                all_scores.extend(scores)
            
            for i, j, score_idx in pair_mappings:
                score = all_scores[score_idx]
                score_matrix[i, j] = score
                score_matrix[j, i] = score
            
            #print("Score Matirx Before ")
            #print(score_matrix)
            assert not np.isnan(score_matrix).any()
            
            #reachables = list()
            #n = len(decoded_seqs)
            
            # for i in range(n): 
            #     row = score_matrix[i]
            #     reachables.append(np.where(row == entailment_score)[0])
            
            # print(reachables)
            
            # for reachable in reachables:
            #     print("reachable", reachable)
            #     for r in reachable: 
            #         connections = reachables[r]
            #         print('connections', connections)
            #         for c in connections: 
            #             if c not in reachable: 
            #                 if score_matrix[r,c] == 1: 
            #                     score_matrix[c, r] = 2
            #                     score_matrix[r, c] = 2
            #                 elif score_matrix[r,c] == 2: 
            #                     connections_tmp = connections[connections != c]
            #                     reachable_tmp = reachable[reachable != r]
            #                     score_matrix[r, connections_tmp] = 1
            #                     score_matrix[c, reachable_tmp] = 1
            

            def enforce_transitive_closure(score_matrix):
                n = score_matrix.shape[0]
                
                # Keep iterating until no changes are made
                changed = True
                iterations = 0
                max_iterations = 100
                
                while changed and iterations < max_iterations:
                    changed = False
                    iterations += 1
                    
                    for i in range(n):
                        for j in range(n):
                            if i == j:
                                continue
                                
                            rel_ij = score_matrix[i, j]
                            if rel_ij == 1:  # neutral, skip
                                continue
                            
                            for k in range(n):
                                if k == i or k == j:
                                    continue
                                
                                rel_jk = score_matrix[j, k]
                                if rel_jk == 1:  # neutral, skip
                                    continue
                                
                                # Calculate what i->k should be based on transitivity
                                if rel_ij == 2 and rel_jk == 2:
                                    expected = 2  # entails + entails = entails
                                elif rel_ij == 2 and rel_jk == 0:
                                    expected = 0  # entails + contradicts = contradicts
                                elif rel_ij == 0 and rel_jk == 2:
                                    expected = 0  # contradicts + entails = contradicts
                                elif rel_ij == 0 and rel_jk == 0:
                                    expected = 0  # contradicts + contradicts = contradicts
                                else:
                                    continue
                                
                                # ONLY update if currently neutral (1)
                                if score_matrix[i, k] == 1:
                                    score_matrix[i, k] = expected
                                    score_matrix[k, i] = expected
                                    changed = True
                                # If there's a conflict, we could log it or handle it
                                # but for now, trust the existing value
                
                return score_matrix

            # Apply
            score_matrix = enforce_transitive_closure(score_matrix)
                            
            #print("Score Matirx After ")
            #print(score_matrix)
            
            cluster_ids = [-1] * len(decoded_seqs)
            topic_ids = [-1] * len(decoded_seqs)
            next_id = 0
            next_topic_id = 0
            
            for i, string1 in enumerate(decoded_seqs):
                # Check if string1 already has an id assigned.
                if cluster_ids[i] == -1:
                    # If string1 has not been assigned an id, assign it next_id.
                    # row = score_matrix[i]
                    #relevant_entries = row[row == 1]
                    #if len(relevant_entries) == len(decoded_seqs)-1: 
                    #    cluster_ids[i] = neutral_placeholder
                    #else: 
                    cluster_ids[i] = next_id
                    for j in range(i+1, len(decoded_seqs)):
                        # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                        # if are_equivalent(string1, strings_list[j]):
                        #    semantic_set_ids[j] = next_id
                        # if score_matrix[i, j] == entailment_score:
                        if score_matrix[i, j] in [2]:
                            cluster_ids[j] = next_id
                    next_id += 1
                    # cluster_ids[i] = next_id
                    # cluster_members = [i]

                    # # Versuche weitere Elemente hinzuzufügen
                    # for j in range(i + 1, n):
                    #     if cluster_ids[j] != -1:
                    #         continue

                    #     # j darf nur rein, wenn es zu ALLEN bisherigen Mitgliedern entailment hat
                    #     ok = True
                    #     for u in cluster_members:
                    #         if score_matrix[u, j] != entailment_score and score_matrix[j, u] != entailment_score:
                    #             ok = False
                    #             break

                    #     if ok:
                    #         cluster_ids[j] = next_id
                    #         cluster_members.append(j)
                    # next_id += 1
                
                if topic_ids[i] == -1: 
                    topic_ids[i] = next_topic_id
                    for j in range(i+1, len(decoded_seqs)):
                        if topic_ids[j] == -1 and score_matrix[i, j] in [contradiction_score, entailment_score]:
                            topic_ids[j] = next_topic_id
                    next_topic_id += 1     

            assert -1 not in cluster_ids
            assert -1 not in topic_ids
            #print('cluster ids after: ', cluster_ids)
            #print('topic ids after: ', topic_ids)
            # cluster_ids = [next_id if x == neutral_placeholder else x for x in cluster_ids]
            # cluster_ids = np.array(cluster_ids)
            # unique_cids = np.unique(cluster_ids)
            # for c_id in unique_cids: 
            #     indices = np.where(cluster_ids == c_id)[0]
            #     cluster_weight = []
            #     for idx in indices: 
            #         row = score_matrix[idx]
            #         row_probs = contr_matrix[idx]
            #         relevant_entries = np.where(row != 2)[0]
            #         if len(relevant_entries) > 0: 
            #             relevant_probs = row_probs[relevant_entries]
            #             cluster_weight.append(relevant_probs.mean())
            #         else: 
            #             cluster_weight.append(1)
                
            #     cluster_mean = sum(cluster_weight) / len(cluster_weight)    
            #     cluster_weights.append(cluster_mean)
                
                    
                # TODO get indices of relevant entries and get corresponding values from row_probs
                    
                # if 1 in all_scores:
                #     indices = np.where(cluster_ids == c_id)[0]
                #     c_weights = []
                #     for idx in indices: 
                #         row = score_matrix[idx]
                #         relevant_entries = row[row == 0]
                #         # Neutral to all other sequences --> drop it
                #         if len(relevant_entries) == 0: 
                #             c_weights.append(0)
                #         else: 
                #             c_weights.append(1) #len(relevant_entries) / len(row))
                #     cluster_weights.append(np.mean(c_weights))
                # # there are no neutral tokens
                # else: 
                #     cluster_weights = [1] * unique_cids
                
                
        cluster_ids_across_steps.append({'cluster_ids' : cluster_ids})
        topic_ids_across_steps.append({'topic_ids' : topic_ids})
        probs_across_steps.append(probs)
        # cluster_weights_across_steps.append({'cluster_weights' : cluster_weights})
        # print(probs)
        # print(cluster_ids)
        # print(topic_ids)
        # print(unique_elements)
        assert len(cluster_ids) == len(unique_elements) == len(probs)
        
        #print(probs)
    
    return cluster_ids_across_steps, topic_ids_across_steps, probs_across_steps
            
        