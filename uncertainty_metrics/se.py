import numpy as np
import nltk

nltk.download('punkt')
            
# Adapted from: https://github.com/jlko/semantic_uncertainty
def logsumexp_by_id(semantic_ids, probs):
    """
    Aggregate probabilities by semantic ID.

    Groups samples by their semantic cluster assignment and sums the
    probabilities within each cluster, normalizing by the total probability
    mass across all clusters.

    Args:
        semantic_ids (list[int]): Per-sample semantic cluster IDs. Must be
            contiguous integers starting from 0 (i.e. [0, 1, ..., K-1]).
        probs (array-like): Per-sample probability values, shape (N,).

    Returns:
        list[float]: Normalized probability mass for each unique semantic ID.
    """
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    probs_per_semantic_id = []
    sum_clusters_probs = np.sum(probs)
    for uid in unique_ids:
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        id_prob = [probs[i] for i in id_indices]
        sum_token_prob = np.sum(id_prob)
        
        prob_norm = sum_token_prob / sum_clusters_probs
        probs_per_semantic_id.append(prob_norm)

    return probs_per_semantic_id

# Adapted from: https://github.com/jlko/semantic_uncertainty
def predictive_entropy_rao(probs):
    if len(probs) == 1: 
        return 0.0
    log_vals = np.where(
            probs == 0,
            np.zeros_like(probs),
            np.log(probs)
        )
    entropy = - np.sum(probs * log_vals)
    return entropy

def compute_se_across_subsequences(cluster_ids_across_steps, probs, topics = None): 
    """
    Compute semantic entropy at each generation step across alternative subsequences.

    For each step, probabilities are aggregated by semantic cluster ID via
    `logsumexp_by_id` and entropy is computed via `predictive_entropy_rao`.

    When `topics` is provided, a claim conditional entropy is estimated.
    Args:
        cluster_ids_across_steps (list[dict]): One dict per step, each with a
            'cluster_ids' key mapping to a list of integer semantic cluster assignments.
        probs (list[list[float]]): Per-step lists of alternative sequence
            probabilities, aligned with `cluster_ids_across_steps`.
        topics (list[dict] | None): Optional list of per-step dicts, each with
            a 'topic_ids' key. If provided, conditional entropy is computed
            instead of raw semantic entropy. Default None.

    Returns:
        list[float]: One entropy (or conditional entropy) value per generation step.
    """
    entropies = []
    if topics is not None: 
        for ids, probs_step, topic in zip(cluster_ids_across_steps, probs, topics): 
            semantic_ids = ids['cluster_ids']
            topic_ids = topic['topic_ids']
            
            probs_per_semantic_id = logsumexp_by_id(semantic_ids, probs=probs_step)
            pe = predictive_entropy_rao(probs_per_semantic_id)
            
            probs_per_claim_id = logsumexp_by_id(topic_ids, probs=probs_step)
            pe_claim = predictive_entropy_rao(probs_per_claim_id)
            cond_pe2 = pe - pe_claim
            entropies.append(cond_pe2)
    else: 
        for ids, probs_step in zip(cluster_ids_across_steps, probs): 
            semantic_ids = ids['cluster_ids']
            probs_per_semantic_id = logsumexp_by_id(semantic_ids, probs=probs_step)
            pe = predictive_entropy_rao(probs_per_semantic_id)
            entropies.append(pe)
    return entropies


def generate_semantic_subsequence_ids(seq_tokens, question, ellm, mode = 'adapted'): 
    """
    Assign semantic cluster IDs and topic IDs to alternative subsequences at each
    generation step using an NLI model.

    For each step in `seq_tokens`, the function:
        1. Extracts the last sentence of each alternative decoded sequence.
        2. Short-circuits single unique sequence by assigning all alternatives to cluster 0.
        3. Otherwise batches all unique sentence pairs through
           `ellm.check_implication_batch` to obtain
           pairwise relation scores:
               - 2 = entailment  (same semantic cluster)
               - 1 = neutral
               - 0 = contradiction
        4. Applies transitive closure over the score matrix so that entailment and contradiction chains are propagated consistently.
        5. Assigns semantic cluster IDs by grouping mutually entailing sequences,
           and topic IDs by grouping sequences that are either entailing or
           contradicting.

    The `mode` parameter controls how the entailment score is set:
        - 'data': entailment_score = 1
        - anything else: entailment_score = 2

    Args:
        seq_tokens (list[dict]): Per-step data dicts, each containing at minimum:
            - 'alternative_sequence_question_decoded' (list[str]): Decoded
              alternative continuations prefixed with the question.
            - 'alternative_token_probs' (list[float]): Probability of each
              alternative token.
        question (str): The prompt string, prepended to decoded sequences before
            entailment classification.
        ellm: Entailment language model with a `check_implication_batch(pairs)`
            method that accepts a list of (str, str) tuples and returns
            (scores, contradiction_probs).
        mode (str): Controls the numeric value used for the entailment relation.
            Use 'data' for value 1, or any other string for value 2. Default 'adapted'.

    Returns:
        tuple:
            list[dict]: Per-step dicts with key 'cluster_ids' (list[int]) —
                semantic cluster assignment for each unique alternative.
            list[dict]: Per-step dicts with key 'topic_ids' (list[int]) —
                topic assignment for each unique alternative.
            list[list[float]]: Per-step lists of unique alternative
                token probabilities.
    """
    cluster_ids_across_steps = []
    # cluster_weights_across_steps = []
    topic_ids_across_steps = []
    probs_across_steps = []
    MAX_BATCH = 32
    
    for s, step in enumerate(seq_tokens): 
        last_sentences = [nltk.sent_tokenize(text)[-1] for text in step['alternative_sequence_question_decoded']]
        decoded_seqs = [question + ' ' + sq for sq in last_sentences]
        probs = step.get('alternative_token_probs', None)        
        
        unique_elements = list(set(zip(decoded_seqs, probs)))
        set_step = set(tuple(sublist) for sublist in decoded_seqs)
        if len(unique_elements) == 1: #or max(probs) > 0.99:
            cluster_ids = [0] * len(unique_elements)
            topic_ids = [0] * len(unique_elements)
            probs = [element[1] for element in unique_elements] #[unique_elements[0][1]]
        else:  
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
            
            all_scores = []
            
            for b in range(0, len(batched_pairs), MAX_BATCH):
                sub = batched_pairs[b:b+MAX_BATCH]
                scores, contr_prob = ellm.check_implication_batch(sub)
                all_scores.extend(scores)
            
            for i, j, score_idx in pair_mappings:
                score = all_scores[score_idx]
                score_matrix[i, j] = score
                score_matrix[j, i] = score
            
            assert not np.isnan(score_matrix).any()

            def enforce_transitive_closure(score_matrix):
                n = score_matrix.shape[0]
                
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
                                
                                if score_matrix[i, k] == 1:
                                    score_matrix[i, k] = expected
                                    score_matrix[k, i] = expected
                                    changed = True
                
                return score_matrix

            score_matrix = enforce_transitive_closure(score_matrix)
            cluster_ids = [-1] * len(decoded_seqs)
            topic_ids = [-1] * len(decoded_seqs)
            next_id = 0
            next_topic_id = 0
            
            for i, string1 in enumerate(decoded_seqs):
                if cluster_ids[i] == -1:
                    cluster_ids[i] = next_id
                    for j in range(i+1, len(decoded_seqs)):
                        if score_matrix[i, j] in [2]:
                            cluster_ids[j] = next_id
                    next_id += 1
                
                if topic_ids[i] == -1: 
                    topic_ids[i] = next_topic_id
                    for j in range(i+1, len(decoded_seqs)):
                        if topic_ids[j] == -1 and score_matrix[i, j] in [contradiction_score, entailment_score]:
                            topic_ids[j] = next_topic_id
                    next_topic_id += 1     

            assert -1 not in cluster_ids
            assert -1 not in topic_ids 
            assert len(cluster_ids) == len(unique_elements) == len(probs)             
                
        cluster_ids_across_steps.append({'cluster_ids' : cluster_ids})
        topic_ids_across_steps.append({'topic_ids' : topic_ids})
        probs_across_steps.append(probs)
        
            
    return cluster_ids_across_steps, topic_ids_across_steps, probs_across_steps
            
        