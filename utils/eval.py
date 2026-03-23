"""Helper functions for evaluating the results. Used in analysis_bios.ipynb and analysis_qa.ipynb"""

from sklearn import metrics
import ast
import numpy as np
import re
import json
import pandas as pd

def auroc(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, drop_intermediate=True)
    del thresholds
    return metrics.auc(fpr, tpr)


def prroc(y_true, y_score):
    return metrics.average_precision_score(y_true, y_score)



def get_entropies(evals_pp, uqs_df, exceptions = [], level = "token"):
    all_entropies = []
    all_ln_probs = []
    for o, output in enumerate(evals_pp):
        question = output['question']
        if o in exceptions: 
            continue
        else: 
            uqs = (uqs_df[question == uqs_df['question']]).to_dict('records')[0] 
            if level == "token": 
                all_ln_probs.extend(uqs['ln_probs_token'])
                all_entropies.extend(uqs['entropies_token'])
            else: 
                all_ln_probs.extend(uqs['ln_probs_word'])
                all_entropies.extend(uqs['entropies_word']) 
    return all_entropies
    


def get_summary_stats(df, pos_list): 
    n = len(df)
    nmb_tokens = 0
    correct_claims = 0
    incorrect_claims = 0
    for res in df: 
        nmb_tokens += res['nmb_tokens']
        if res['is_supported']: 
            correct_claims += 1
        else: incorrect_claims += 1

    correct_tokens = 0
    incorrect_tokens = 0    
    for el in pos_list: 
        correct_tokens += len(el['ses_no']) 
        incorrect_tokens += len(el['ses_yes'])
    
    return nmb_tokens / n, correct_claims, incorrect_claims, correct_tokens, incorrect_tokens


def get_filtered_results(evals_pp, uqs_df, exceptions = [], metric = "roc", level = "token", boundaries = [50, 55, 60, 65, 70, 75, 100], decreasing = False):
    all_entropies = []
    all_ln_probs = []
    results_across_quartiles_claim_lvl = []
    results_across_quartiles_instance_lvl = []
    
    
    for o, output in enumerate(evals_pp):
        question = output['question']
        if o in exceptions: 
            continue
        else: 
            uqs = (uqs_df[question == uqs_df['question']]).to_dict('records')[0] 
            if level == "token": 
                all_ln_probs.extend(uqs['ln_probs_token'])
                all_entropies.extend(uqs['entropies_token'])
            else: 
                all_ln_probs.extend(uqs['ln_probs_word'])
                all_entropies.extend(uqs['entropies_word'])
    
    # quartile_boundaries = np.percentile(all_entropies, boundaries)
    quartile_boundaries = [0.1, 0.5, 1, 2, 3, 7] 
    if level == 'token': 
        res, pos_perf = get_results_token_lvl(evals_pp, uqs_df, exceptions, do_filter = False)
    else: 
        res, pos_perf = get_results_word_lvl(evals_pp, uqs_df, exceptions, do_filter = False)
        
    avg_nmb_tokens_all, correct_claims_all, incorrect_claims_all, correct_tokens_all, incorrect_tokens_all = get_summary_stats(res, pos_perf)
    for b, boundary in enumerate(quartile_boundaries): 
        if not decreasing:
            if b > 0: 
                if level == 'token': 
                    res, pos_perf = get_results_token_lvl(evals_pp, uqs_df, exceptions, do_filter = True, upper_threshold = boundary, lower_threshold=quartile_boundaries[b-1])
                else: 
                    res, pos_perf = get_results_word_lvl(evals_pp, uqs_df, exceptions, do_filter = True, upper_threshold = boundary, lower_threshold=quartile_boundaries[b-1])
            else: 
                if level == 'token': 
                    res, pos_perf = get_results_token_lvl(evals_pp, uqs_df, exceptions, do_filter = True, upper_threshold = boundary, lower_threshold=-0.00000001)
                else: 
                    res, pos_perf = get_results_word_lvl(evals_pp, uqs_df, exceptions, do_filter = True, upper_threshold = boundary, lower_threshold=-0.0000001)
        else: 
            if boundary == 7: 
                break
            if b > 0:
                if level == 'token': 
                    res, pos_perf = get_results_token_lvl(evals_pp, uqs_df, exceptions, do_filter = True, upper_threshold = quartile_boundaries[-1], lower_threshold=quartile_boundaries[b])
                else: 
                    res, pos_perf = get_results_word_lvl(evals_pp, uqs_df, exceptions, do_filter = True, upper_threshold = quartile_boundaries[-1], lower_threshold=quartile_boundaries[b])
            else: 
                if level == 'token': 
                    res, pos_perf = get_results_token_lvl(evals_pp, uqs_df, exceptions, do_filter = True, upper_threshold = quartile_boundaries[-1], lower_threshold=-0.00000001)
                else: 
                    res, pos_perf = get_results_word_lvl(evals_pp, uqs_df, exceptions, do_filter = True, upper_threshold = quartile_boundaries[-1], lower_threshold=-0.0000001)
            
        avg_nmb_tokens, correct_claims, incorrect_claims, correct_tokens, incorrect_tokens = get_summary_stats(res, pos_perf)        
        res = pd.DataFrame(res)  
        res['label'] = (~res['is_supported']).astype(int) 
        
        if level == 'token': 
            mean_stds = res['stds']
            mean_stds_false = [p['stds_yes'] for p in pos_perf]
            mean_stds_true = [p['stds_no'] for p in pos_perf]
            examples_true = [p['examples_no'] for p in pos_perf]
            examples_false = [p['examples_yes'] for p in pos_perf]
            lns_true = [p['ln_no'] for p in pos_perf]
            lns_false = [p['ln_yes'] for p in pos_perf]
            en_true = [p['e_no'] for p in pos_perf]
            en_false = [p['e_yes'] for p in pos_perf]
            
        else: 
            mean_stds = None
            mean_stds_false = None
            mean_stds_true = None
            examples_true, examples_false, lns_true, lns_false, en_true, en_false = None, None, None, None, None, None
        
        auc_ln_probs, auc_entropy, auc_se, auc_se_w, auc_vnes_emb_disp, auc_vnes_emb, auc_vnes_emb_disp_rbf, auc_vnes_emb_rbf = get_perf(res, comp = metric, print_it=False)
        # print(boundary, auc_entropy)
        results_across_quartiles_claim_lvl.append({
            "percentile" : boundary, 
            "auc_ln_probs" : auc_ln_probs, 
            "auc_entropy" : auc_entropy, 
            "auc_se" : auc_se,
            "auc_se_w" : auc_se_w,
            "auc_se_w" : auc_se_w, 
            "auc_vnes_emb" : auc_vnes_emb, 
            "auc_vnes_emb_disp" : auc_vnes_emb_disp, 
            "auc_vnes_emb_rbf" : auc_vnes_emb_rbf, 
            "auc_vnes_emb_disp_rbf" : auc_vnes_emb_disp_rbf, 
            "avg_nmb_tokens" : avg_nmb_tokens, 
            "correct_claims" : correct_claims, 
            "incorrect_claims" : incorrect_claims, 
            "correct_tokens" : correct_tokens,
            "incorrect_tokens" : incorrect_tokens,
            "correct_claims_all" : correct_claims_all, 
            "incorrect_claims_all" : incorrect_claims_all, 
            "correct_tokens_all" : correct_tokens_all,
            "incorrect_tokens_all" : incorrect_tokens_all
        }) 
        vnes_values, vnes_values_disp, vnes_values_rbf, vnes_values_disp_rbf, ses_values, ses_w_values, ln_values, e_values, labels = get_position_values(pos_perf)
        auc_ln_probs, auc_entropy, auc_se, auc_se_w, auc_vnes_emb_disp, auc_vnes_emb, auc_vnes_emb_disp_rbf, auc_vnes_emb_rbf = get_position_perf(vnes_values, vnes_values_disp, vnes_values_rbf, vnes_values_disp_rbf, ses_values, ses_w_values, ln_values, e_values, labels, metric = metric, print_it=False)   
        results_across_quartiles_instance_lvl.append({
            "percentile" : boundary, 
            "auc_ln_probs" : auc_ln_probs, 
            "auc_entropy" : auc_entropy, 
            "auc_se" : auc_se,
            "auc_se_w" : auc_se_w,
            "auc_se_w" : auc_se_w, 
            "auc_vnes_emb" : auc_vnes_emb, 
            "auc_vnes_emb_disp" : auc_vnes_emb_disp, 
            "auc_vnes_emb_rbf" : auc_vnes_emb_rbf, 
            "auc_vnes_emb_disp_rbf" : auc_vnes_emb_disp_rbf, 
            "avg_nmb_tokens" : avg_nmb_tokens, 
            "correct_claims" : correct_claims, 
            "incorrect_claims" : incorrect_claims, 
            "correct_tokens" : correct_tokens,
            "incorrect_tokens" : incorrect_tokens,
            "correct_claims_all" : correct_claims_all, 
            "incorrect_claims_all" : incorrect_claims_all, 
            "correct_tokens_all" : correct_tokens_all,
            "incorrect_tokens_all" : incorrect_tokens_all, 
            'mean_stds' : mean_stds, 
            'mean_stds_true' : mean_stds_true, 
            'mean_stds_false' : mean_stds_false,
            'examples_true' : examples_true, 
            'examples_false' : examples_false, 
            'lns_true' : lns_true, 
            'lns_false' : lns_false, 
            'en_true' : en_true,
            'en_false' : en_false
        }) 
    return results_across_quartiles_claim_lvl, results_across_quartiles_instance_lvl  


def get_results_token_lvl(evals_pp, uqs_df, exceptions = [], do_filter = False, upper_threshold = None, lower_threshold=None): 
    results = []
    
    position_performance = []
   
    
    for o, output in enumerate(evals_pp):
        if o in exceptions: 
            continue
        position_lists = []
        question = output['question']
        generated_text = output['gen_text']
        uqs = (uqs_df[question == uqs_df['question']]).to_dict('records')[0] 
        for f, fact in enumerate(output['acc_facts']):
            try: 
                indices = ast.literal_eval(fact['matched_token_indices'])
            except:
                indices = fact['matched_token_indices']
        
            if len(indices) == 0: 
                continue
            gen_tokens = uqs['gen_tokens']
            try: 
                try: 
                    try:
                        acc_words = ast.literal_eval(fact['acc_tokens'])
                    except Exception as e: 
                        print(e)
                        acc_words = json.loads(fact['acc_tokens'])
                except Exception as e:
                    print(e)
                    acc_words = fact['acc_tokens']
                if isinstance(acc_words, dict): 
                    if acc_words.get('mappings', None) is not None: 
                        acc_words = [item.get('value') for item in acc_words['mappings']]
                    else: 
                        acc_words = [item.get('value') for item in acc_words]
                elif isinstance(acc_words[0], dict): 
                    acc_words = [item.get('value') for item in acc_words]
                if len(acc_words) != len(uqs['ses_tokens_to']):
                    continue
                else: 
                    position_lists.append(acc_words)
            except Exception as e: 
                print('fact ', f, fact['fact'])
                print(e)
                continue

            if do_filter:                 
                matched_se_word = [uqs['ses_tokens_to'][i] for i in indices if lower_threshold < uqs['entropies_token'][i] <= upper_threshold]
                matched_se_word_w = [uqs['ses_tokens_to_w'][i] for i in indices if lower_threshold < uqs['entropies_token'][i] <= upper_threshold]            
                entropies = [uqs['entropies_token'][i] for i in indices if lower_threshold < uqs['entropies_token'][i] <= upper_threshold]
                ln_probs_word = [uqs['ln_probs_token'][i] for i in indices if lower_threshold < uqs['entropies_token'][i] <= upper_threshold]
                
                vnes_word_emb = [uqs['vnes_token'][i] for i in indices if lower_threshold < uqs['entropies_token'][i] <= upper_threshold]
                vnes_word_emb_rbf = [uqs['vnes_token_rbf'][i] for i in indices if lower_threshold < uqs['entropies_token'][i] <= upper_threshold]
                vnes_word_multpl_combined = [uqs['vnes_token_multpl_combined'][i] for i in indices if lower_threshold < uqs['entropies_token'][i] <= upper_threshold]
                vnes_word_add_combined = [uqs['vnes_token_add_combined'][i] for i in indices if lower_threshold < uqs['entropies_token'][i] <= upper_threshold]
                vnes_word_word = [uqs['vnes_token_token'][i] for i in indices if lower_threshold < uqs['entropies_token'][i] <= upper_threshold]
                
                vnes_word_emb_disp = [uqs['vnes_token_disp'][i] for i in indices if lower_threshold < uqs['entropies_token'][i] <= upper_threshold]
                vnes_word_emb_disp_rbf = [uqs['vnes_token_disp_rbf'][i] for i in indices if lower_threshold < uqs['entropies_token'][i] <= upper_threshold]
                vnes_word_multpl_combined_disp  = [uqs['vnes_token_multpl_combined_disp'][i] for i in indices if lower_threshold < uqs['entropies_token'][i] <= upper_threshold]
                vnes_word_add_combined_disp  = [uqs['vnes_token_add_combined_disp'][i] for i in indices if lower_threshold < uqs['entropies_token'][i] <= upper_threshold]
                vnes_word_word_disp  = [uqs['vnes_token_token_disp'][i] for i in indices if lower_threshold < uqs['entropies_token'][i] <= upper_threshold]
                stds = [std for i, std in enumerate(uqs['std_emb_token']) if lower_threshold < uqs['entropies_token'][i] <= upper_threshold]
            else: 
                vnes_word_emb = [uqs['vnes_token'][i] for i in indices]
                vnes_word_emb_rbf = [uqs['vnes_token_rbf'][i] for i in indices]
                
                matched_se_word = [uqs['ses_tokens_to'][i] for i in indices]
                matched_se_word_w = [uqs['ses_tokens_to_w'][i] for i in indices]            
                entropies = [uqs['entropies_token'][i] for i in indices]
                ln_probs_word = [uqs['ln_probs_token'][i] for i in indices]
                
                vnes_word_multpl_combined = [uqs['vnes_token_multpl_combined'][i] for i in indices]
                vnes_word_add_combined = [uqs['vnes_token_add_combined'][i] for i in indices]
                vnes_word_word = [uqs['vnes_token_token'][i] for i in indices]
                
                vnes_word_emb_disp = [uqs['vnes_token_disp'][i] for i in indices]
                vnes_word_emb_disp_rbf = [uqs['vnes_token_disp_rbf'][i] for i in indices]
                
                vnes_word_multpl_combined_disp  = [uqs['vnes_token_multpl_combined_disp'][i] for i in indices]
                vnes_word_add_combined_disp  = [uqs['vnes_token_add_combined_disp'][i] for i in indices]
                vnes_word_word_disp  = [uqs['vnes_token_token_disp'][i] for i in indices]
                stds = uqs['std_emb_token']

            if len(ln_probs_word) > 0: 
                results.append({'o' : o,
                                'question' : question,
                                'is_supported' : fact['supported'], 
                                'matched_se' : matched_se_word,
                                'matched_se_w' : matched_se_word_w, 
                                'vnes_add_combined' : vnes_word_add_combined, 
                                'vnes_emb' : vnes_word_emb, 
                                'vnes_emb_rbf' : vnes_word_emb_rbf, 
                                'vnes_multpl_combined' : vnes_word_multpl_combined, 
                                'vnes_word_add_combined' : vnes_word_add_combined, 
                                'vnes_element' : vnes_word_word, 
                                # -------------------------
                                'vnes_add_combined_disp' : vnes_word_add_combined_disp, 
                                'vnes_emb_disp' : vnes_word_emb_disp,
                                'vnes_emb_disp_rbf' : vnes_word_emb_disp_rbf, 
                                'vnes_multpl_combined_disp' : vnes_word_multpl_combined_disp, 
                                'vnes_word_add_combined_disp' : vnes_word_add_combined_disp, 
                                'vnes_element_disp' : vnes_word_word_disp, 
                                # --------------------------
                                'ln_probs' : ln_probs_word,
                                'entropies' : entropies, 
                                'fact_rank' : f, 
                                'len_text'  : len(generated_text),
                                'nmb_tokens' : len(gen_tokens),
                                'stds' : stds
                                })

        
        # ------------------- list iof lists verarbeiten 
        collected_positions = [
            any(v == "yes" for v in values)
            for values in zip(*position_lists)
        ]

        if do_filter: 
            yes_indices = [i for i, v in enumerate(collected_positions) if v and lower_threshold < uqs['entropies_token'][i] <= upper_threshold]
            no_indices  = [i for i, v in enumerate(collected_positions) if not v and lower_threshold < uqs['entropies_token'][i] <= upper_threshold]
        else: 
            yes_indices = [i for i, v in enumerate(collected_positions) if v]
            no_indices  = [i for i, v in enumerate(collected_positions) if not v]
        
        try:
            gen_tokens= ast.literal_eval(output['gen_tokens'])
        except:
            gen_tokens= output['gen_tokens']

        position_performance.append({'o' : o, 
                                     'question' : question, 
                                     'vnes_yes' : [uqs['vnes_token'][i] for i in yes_indices],
                                     'vnes_no' : [uqs['vnes_token'][i] for i in no_indices],
                                     'vnes_yes_disp' : [uqs['vnes_token_disp'][i] for i in yes_indices],
                                     'vnes_no_disp' : [uqs['vnes_token_disp'][i] for i in no_indices],
                                     'vnes_yes_rbf' : [uqs['vnes_token_rbf'][i] for i in yes_indices],
                                     'vnes_no_rbf' : [uqs['vnes_token_rbf'][i] for i in no_indices],
                                     'vnes_yes_disp_rbf' : [uqs['vnes_token_disp_rbf'][i] for i in yes_indices],
                                     'vnes_no_disp_rbf' : [uqs['vnes_token_disp_rbf'][i] for i in no_indices],
                                     'ses_yes' : [uqs['ses_tokens_to'][i] for i in yes_indices],
                                     'ses_no' : [uqs['ses_tokens_to'][i] for i in no_indices],
                                     'ses_w_yes' : [uqs['ses_tokens_to_w'][i] for i in yes_indices],
                                     'ses_w_no' : [uqs['ses_tokens_to_w'][i] for i in no_indices],
                                     'e_yes' : [uqs['entropies_token'][i] for i in yes_indices],
                                     'e_no' : [uqs['entropies_token'][i] for i in no_indices],
                                     'ln_yes' : [uqs['ln_probs_token'][i] for i in yes_indices],
                                     'ln_no' : [uqs['ln_probs_token'][i] for i in no_indices],
                                     'stds_yes' : [uqs['std_emb_token'][i] for i in yes_indices],
                                     'stds_no' : [uqs['std_emb_token'][i] for i in no_indices],
                                     'examples_no' : [uqs['seq_tokens'][i]['alternative_tokens_str'] for i in no_indices],
                                     'examples_yes' : [uqs['seq_tokens'][i]['alternative_tokens_str'] for i in yes_indices],
                                     
                                     })
        
        
    return results, position_performance 
def get_results_word_lvl(evals_pp, uqs_df, exceptions = [], do_filter = True, upper_threshold = None, lower_threshold=None): 
    results = []
    position_performance = []
    
    
    for o, output in enumerate(evals_pp):
        position_lists = []
        if o in exceptions: 
            continue
        question = output['question']
        generated_text = output['gen_text']
        gen_words = ast.literal_eval(output['gen_words_x'])
        for f, fact in enumerate(output['acc_facts']):
            indices = ast.literal_eval(fact['matched indices'])
            uqs = (uqs_df[question == uqs_df['question']]).to_dict('records')[0] 
            if len(indices) == 0: 
                continue
            
            try: 
                try: 
                    try:
                        acc_words = ast.literal_eval(fact['acc_words'])
                    except Exception as e: 
                        print(e)
                        acc_words = json.loads(fact['acc_words'])
                except Exception as e:
                    print(e)
                    acc_words = fact['acc_words']
                if isinstance(acc_words, dict): 
                    if acc_words.get('mappings') is not None: 
                        acc_words = [item.get('value') for item in acc_words['mappings']]
                    else: 
                        acc_words = [item.get('value') for item in acc_words]
                elif isinstance(acc_words[0], dict): 
                    acc_words = [item.get('value') for item in acc_words] 
                if len(acc_words) != len(uqs['vnes_word_emb']):
                    continue
                else: 
                    position_lists.append(acc_words)
            except Exception as e: 
                print(e)
                continue
            
            
            
            if not do_filter:  
                ln_probs_word = [uqs['ln_probs_word'][i] for i in indices]
                entropies = [uqs['entropies_word'][i] for i in indices]
                matched_se_word = [uqs['ses_word_to'][i] for i in indices]
                matched_se_word_w = [uqs['ses_word_to_w'][i] for i in indices]              
                vnes_word_emb = [uqs['vnes_word_emb'][i] for i in indices]
                vnes_word_emb_rbf = [uqs['vnes_word_emb_rbf'][i] for i in indices]

                vnes_word_multpl_combined = [uqs['vnes_word_multpl_combined'][i] for i in indices]
                vnes_word_add_combined = [-uqs['vnes_word_add_combined'][i] for i in indices]
                vnes_word_word = [uqs['vnes_word_word'][i] for i in indices]
                
                
                vnes_word_emb_disp = [uqs['vnes_word_emb_disp'][i] for i in indices]
                vnes_word_emb_disp_rbf = [uqs['vnes_word_emb_disp_rbf'][i] for i in indices]
                
                vnes_word_multpl_combined_disp  = [uqs['vnes_word_multpl_combined_disp'][i] for i in indices]
                vnes_word_add_combined_disp  = [uqs['vnes_word_add_combined_disp'][i] for i in indices]
                vnes_word_word_disp  = [uqs['vnes_word_word_disp'][i] for i in indices]

            else: 
                ln_probs_word = [uqs['ln_probs_word'][i] for i in indices if lower_threshold < uqs['entropies_word'][i] <= upper_threshold]
                entropies = [uqs['entropies_word'][i] for i in indices if lower_threshold < uqs['entropies_word'][i] <= upper_threshold]
                matched_se_word = [uqs['ses_word_to'][i] for i in indices if lower_threshold < uqs['entropies_word'][i] <= upper_threshold]
                matched_se_word_w = [uqs['ses_word_to_w'][i] for i in indices if lower_threshold < uqs['entropies_word'][i] <= upper_threshold]          
                vnes_word_emb = [uqs['vnes_word_emb'][i] for i in indices if lower_threshold < uqs['entropies_word'][i] <= upper_threshold]
                vnes_word_emb_rbf = [uqs['vnes_word_emb_rbf'][i] for i in indices if lower_threshold < uqs['entropies_word'][i] <= upper_threshold]

                vnes_word_multpl_combined = [uqs['vnes_word_multpl_combined'][i] for i in indices if lower_threshold < uqs['entropies_word'][i] <= upper_threshold]
                vnes_word_add_combined = [-uqs['vnes_word_add_combined'][i] for i in indices if lower_threshold < uqs['entropies_word'][i] <= upper_threshold]
                vnes_word_word = [uqs['vnes_word_word'][i] for i in indices if lower_threshold < uqs['entropies_word'][i] <= upper_threshold]
                
                vnes_word_emb_disp = [uqs['vnes_word_emb_disp'][i] for i in indices if lower_threshold < uqs['entropies_word'][i] <= upper_threshold]
                vnes_word_emb_disp_rbf = [uqs['vnes_word_emb_disp_rbf'][i] for i in indices if lower_threshold < uqs['entropies_word'][i] <= upper_threshold]
                
                vnes_word_multpl_combined_disp  = [uqs['vnes_word_multpl_combined_disp'][i] for i in indices if lower_threshold < uqs['entropies_word'][i] <= upper_threshold]
                vnes_word_add_combined_disp  = [uqs['vnes_word_add_combined_disp'][i] for i in indices if lower_threshold < uqs['entropies_word'][i] <= upper_threshold]
                vnes_word_word_disp  = [uqs['vnes_word_word_disp'][i] for i in indices if lower_threshold < uqs['entropies_word'][i] <= upper_threshold]

            
            stds = uqs['std_emb_word']
            
            results.append({'o' : o,
                            'question' : question,
                            'is_supported' : fact['supported'], 
                            'matched_se' : matched_se_word,
                            'matched_se_w' : matched_se_word_w, 
                            'vnes_add_combined' : vnes_word_add_combined, 
                            'vnes_emb' : vnes_word_emb, 
                            'vnes_emb_rbf' : vnes_word_emb_rbf, 
                            
                            #'vnes_emb_ct' : vnes_word_emb_ct, 
                            'vnes_multpl_combined' : vnes_word_multpl_combined, 
                            'vnes_word_add_combined' : vnes_word_add_combined, 
                            'vnes_element' : vnes_word_word, 
                            # 'vnes_deltas' : vnes_word_deltas,
                            # 'raos' : word_rao, 
                            # 'conflict' : word_conflict, 
                            # 'conflict_ct' : word_conflict_ct,
                            # -------------------------
                            'vnes_add_combined_disp' : vnes_word_add_combined_disp, 
                            'vnes_emb_disp' : vnes_word_emb_disp, 
                            'vnes_emb_disp_rbf' : vnes_word_emb_disp_rbf, 
                            
                            'vnes_multpl_combined_disp' : vnes_word_multpl_combined_disp, 
                            'vnes_word_add_combined_disp' : vnes_word_add_combined_disp, 
                            'vnes_element_disp' : vnes_word_word_disp, 
                            # 'raos_disp' : word_rao_disp, 
                            # 'conflict_disp' : word_conflict_disp,
                            # --------------------------
                            'ln_probs' : ln_probs_word,
                            'entropies' : entropies, 
                            'fact_rank' : f, 
                            'len_text'  : len(generated_text),
                            'nmb_tokens' : len(gen_words),
                            'stds' : stds
                            })
            
        # ------------------- list iof lists verarbeiten 
        #print([len(e) for e in position_lists])
        # print(position_lists)
        collected_positions = [
            any(v == "yes" for v in values)
            for values in zip(*position_lists)
        ]
        if do_filter: 
            try: 
                yes_indices = [i for i, v in enumerate(collected_positions) if v and lower_threshold < uqs['entropies_word'][i] <= upper_threshold]
                no_indices  = [i for i, v in enumerate(collected_positions) if not v and lower_threshold < uqs['entropies_word'][i] <= upper_threshold]
            except: 
                yes_indices = [i for i, v in enumerate(collected_positions) if v]
                no_indices  = [i for i, v in enumerate(collected_positions) if not v]
        else: 
            yes_indices = [i for i, v in enumerate(collected_positions) if v]
            no_indices  = [i for i, v in enumerate(collected_positions) if not v]
        
        position_performance.append({'o' : o, 
                                     'question' : question, 
                                     'vnes_yes' : [uqs['vnes_word_emb'][i] for i in yes_indices],
                                     'vnes_no' : [uqs['vnes_word_emb'][i] for i in no_indices],
                                     'vnes_yes_disp' : [uqs['vnes_word_emb_disp'][i] for i in yes_indices],
                                     'vnes_no_disp' : [uqs['vnes_word_emb_disp'][i] for i in no_indices],
                                     'vnes_yes_rbf' : [uqs['vnes_word_emb_rbf'][i] for i in yes_indices],
                                     'vnes_no_rbf' : [uqs['vnes_word_emb_rbf'][i] for i in no_indices],
                                     'vnes_yes_disp_rbf' : [uqs['vnes_word_emb_disp_rbf'][i] for i in yes_indices],
                                     'vnes_no_disp_rbf' : [uqs['vnes_word_emb_disp_rbf'][i] for i in no_indices],
                                     'ses_yes' : [uqs['ses_word_to'][i] for i in yes_indices],
                                     'ses_no' : [uqs['ses_word_to'][i] for i in no_indices],
                                     'ses_w_yes' : [uqs['ses_word_to_w'][i] for i in yes_indices],
                                     'ses_w_no' : [uqs['ses_word_to_w'][i] for i in no_indices],
                                     'e_yes' : [uqs['entropies_word'][i] for i in yes_indices],
                                     'e_no' : [uqs['entropies_word'][i] for i in no_indices],
                                     'ln_yes' : [uqs['ln_probs_word'][i] for i in yes_indices],
                                     'ln_no' : [uqs['ln_probs_word'][i] for i in no_indices],
                                     })

    return results, position_performance

def only_non_zero(elements): 
    no_zeros = []
    for x in elements: 
        if x != 0:
            no_zeros.append(x)
    
    if len(no_zeros) == 0: 
        return 0
    else: 
        return np.nanmean(no_zeros)
        
def get_perf(df, comp = 'roc', print_it = False):
    cols = [
        'matched_se', 'matched_se_w', 'vnes_emb', 'vnes_emb_rbf',
        'vnes_multpl_combined', 'ln_probs', 'entropies', 'vnes_emb_disp',
        'vnes_emb_disp_rbf', 'vnes_multpl_combined_disp'
    ]
    df = df[df[cols].apply(lambda col: col.map(lambda x: len(x) > 0)).all(axis=1)]
    if comp == 'pr':
        auc_se = prroc(df['label'],  df['matched_se'].apply(np.nanmean))
        auc_se_w = prroc(df['label'],  df['matched_se_w'].apply(np.nanmean))
        # ---------------------------
        auc_vnes_emb = prroc(df['label'],  df['vnes_emb'].apply(np.nanmean))
        auc_vnes_emb_rbf = prroc(df['label'],  df['vnes_emb_rbf'].apply(np.nanmean))
        auc_vnes_multpl_combined = prroc(df['label'],  df['vnes_multpl_combined'].apply(np.nanmean))
        # ---------------------------
        auc_ln_probs = prroc(df['label'],  df['ln_probs'].apply(np.nanmean))
        auc_entropy = prroc(df['label'],  df['entropies'].apply(np.nanmean))
        auc_vnes_emb_disp = prroc(df['label'],  df['vnes_emb_disp'].apply(np.nanmean))
        auc_vnes_emb_disp_rbf = prroc(df['label'],  df['vnes_emb_disp_rbf'].apply(np.nanmean))
        auc_vnes_multpl_combined_disp = prroc(df['label'],  df['vnes_multpl_combined_disp'].apply(np.nanmean))
    else: 
        auc_se = auroc(df['label'],  df['matched_se'].apply(np.nanmean))
        auc_se_w = auroc(df['label'],  df['matched_se_w'].apply(np.nanmean))
        auc_vnes_emb = auroc(df['label'],  df['vnes_emb'].apply(np.nanmean))
        auc_vnes_emb_rbf = auroc(df['label'],  df['vnes_emb_rbf'].apply(np.nanmean))
        auc_vnes_multpl_combined = auroc(df['label'],  df['vnes_multpl_combined'].apply(np.nanmean))
        auc_ln_probs = auroc(df['label'],  df['ln_probs'].apply(np.nanmean))
        auc_entropy = auroc(df['label'],  df['entropies'].apply(np.nanmean))
        auc_vnes_emb_disp = auroc(df['label'],  df['vnes_emb_disp'].apply(np.nanmean))
        auc_vnes_emb_disp_rbf = auroc(df['label'],  df['vnes_emb_disp_rbf'].apply(np.nanmean))
        auc_vnes_multpl_combined_disp = auroc(df['label'],  df['vnes_multpl_combined_disp'].apply(np.nanmean))
       
    if print_it: 
        print("auc_se", np.round(auc_se, 4))
        print("auc_se_w", np.round(auc_se_w, 4))
        print("auc_vnes_emb", np.round(auc_vnes_emb, 4))
        print("auc_vnes_multpl_combined", np.round(auc_vnes_multpl_combined, 4))
        print("auc_vnes_emb_disp", np.round(auc_vnes_emb_disp, 4))
        print("auc_vnes_multpl_combined_disp", np.round(auc_vnes_multpl_combined_disp, 4))
        print("auc_ln_probs", np.round(auc_ln_probs, 4))
        print("auc_entropy", np.round(auc_entropy, 4))  
    return auc_ln_probs, auc_entropy, auc_se, auc_se_w, auc_vnes_emb_disp, auc_vnes_emb, auc_vnes_emb_rbf, auc_vnes_emb_disp_rbf

def get_position_values(pos_list):
    vnes_pos, vnes_neg, vnes_values = [], [], []
    vnes_pos_disp, vnes_neg_disp, vnes_values_disp = [], [], []
    vnes_pos_rbf, vnes_neg_rbf, vnes_values_rbf = [], [], []
    vnes_pos_disp_rbf, vnes_neg_disp_rbf, vnes_values_disp_rbf = [], [], []
    
    ses_pos, ses_neg, ses_values = [], [], []
    ses_pos_w, ses_neg_w, ses_w_values = [], [], []
    ln_pos, ln_neg, ln_values = [], [], []
    e_pos, e_neg, e_values = [], [], []
    labels = []
    
    for e, element in enumerate(pos_list):
        vnes_pos.extend([element['vnes_yes']])
        vnes_neg.extend(element['vnes_no'])
        vnes_values.extend(element['vnes_yes'] + element['vnes_no'])
        
        vnes_pos_disp.extend([element['vnes_yes_disp']])
        vnes_neg_disp.extend(element['vnes_no_disp'])
        vnes_values_disp.extend(element['vnes_yes_disp'] + element['vnes_no_disp'])
        
        vnes_pos_rbf.extend([element['vnes_yes_rbf']])
        vnes_neg_rbf.extend(element['vnes_no_rbf'])
        vnes_values_rbf.extend(element['vnes_yes_rbf'] + element['vnes_no_rbf'])
        
        vnes_pos_disp_rbf.extend([element['vnes_yes_disp_rbf']])
        vnes_neg_disp_rbf.extend(element['vnes_no_disp_rbf'])
        vnes_values_disp_rbf.extend(element['vnes_yes_disp_rbf'] + element['vnes_no_disp_rbf'])
        
        ses_pos.extend(element['ses_yes'])
        ses_neg.extend(element['ses_no'])
        ses_values.extend(element['ses_yes'] + element['ses_no'])
        
        ses_pos_w.extend(element['ses_w_yes'])
        ses_neg_w.extend(element['ses_w_no'])
        ses_w_values.extend(element['ses_w_yes'] + element['ses_w_no'])
        
        ln_pos.extend(element['ln_yes'])
        ln_neg.extend(element['ln_no'])
        ln_values.extend(element['ln_yes'] + element['ln_no'])
        
        e_pos.extend(element['e_yes'])
        e_neg.extend(element['e_no'])
        e_values.extend(element['e_yes'] + element['e_no'])
        
        labels.extend(len(element['vnes_yes']) * [1] + len(element['vnes_no']) * [0])
    
    return vnes_values, vnes_values_disp, vnes_values_rbf, vnes_values_disp_rbf, ses_values, ses_w_values, ln_values, e_values, labels       


# ── Struktur ──────────────────────────────────────────────────────────────────
MODELS   = ["Mistral-7B", "Qwen3-4B", "Phi-4-mini"]
METRICS  = ["Neg. log-Prob.", "Entropy", "SE (disp.)", "SE (contr.)",
            "VNE (disp., cosine)", "VNE (contr., cosine)",
            "VNE (disp., rbf)", "VNE (contr., rbf)"]
METHODS  = ["Sampling", "Selection"]

def init_task_dict() -> dict:
    """
    Erstellt ein leeres task_dict mit None für alle Einträge.
    Struktur: task_dict[model][metric][method] = float | None
    """
    return {
        model: {
            metric: {method: None for method in METHODS}
            for metric in METRICS
        }
        for model in MODELS
    }


def update_task_dict(
    task_dict:          dict,
    model:              str,
    method:             str,   # "Sampling" oder "Selection"
    auc_ln_probs:       float,
    auc_entropy:        float,
    auc_se:             float,
    auc_se_w:           float,
    auc_vne_disp_cos:   float,
    auc_vne_contr_cos:  float,
    auc_vne_disp_rbf:   float,
    auc_vne_contr_rbf:  float,
) -> dict:
    values = {
        "Neg. log-Prob.":       auc_ln_probs,
        "Entropy":              auc_entropy,
        "SE (disp.)":           auc_se,
        "SE (contr.)":          auc_se_w,
        "VNE (disp., cosine)":  auc_vne_disp_cos,
        "VNE (contr., cosine)": auc_vne_contr_cos,
        "VNE (disp., rbf)":     auc_vne_disp_rbf,
        "VNE (contr., rbf)":    auc_vne_contr_rbf,
    }

    for metric, val in values.items():
        task_dict[model][metric][method] = val

    return task_dict


def fill_latex_table(template: str, task_dict: dict, caption: str) -> str:
    lines = template.split('\n')
    result_lines = []

    for line in lines:
        #matched = False
        if re.match(r'\s*\\caption\{', line):
            line = f'\\caption{{{caption}}}'
        for metric in METRICS:
            escaped = re.escape(metric)
            if re.match(rf'\s*{escaped}\s*&', line):
                parts = line.split('&')
                new_parts = [parts[0]]  # Zeilenname behalten

                col = 1
                for model in MODELS:
                    for method in METHODS:
                        val = task_dict[model][metric][method]
                        if val is None:
                            new_parts.append(parts[col] if col < len(parts) else '  ')
                        else:
                            new_parts.append(f'  {val * 100:.2f}')
                        col += 1

                # Abschließendes "\\" anhängen
                new_parts.append(parts[-1])
                line = '&'.join(new_parts)
                #matched = True
                break

        result_lines.append(line)

    return '\n'.join(result_lines)


def get_and_write_performance(task_dict, df_cos, metric = "roc", model = "Mistral-7B", method = "Sampling", print_it = False): 
    auc_ln_probs, auc_entropy, auc_se, auc_se_w, auc_vnes_emb_disp, auc_vnes_emb, auc_vnes_emb_rbf, auc_vnes_emb_disp_rbf = get_perf(df_cos, comp = metric, print_it=print_it)
    task_dict = update_task_dict(
        task_dict, model=model, method=method,
        auc_ln_probs=auc_ln_probs, auc_entropy=auc_entropy,
        auc_se=auc_se,      auc_se_w=auc_se_w,
        auc_vne_disp_cos=auc_vnes_emb_disp, auc_vne_contr_cos=auc_vnes_emb,
        auc_vne_disp_rbf=auc_vnes_emb_disp_rbf,  auc_vne_contr_rbf=auc_vnes_emb_rbf,
    )
    return task_dict


def get_position_perf(vnes_values, vnes_values_disp, vnes_values_rbf, vnes_values_disp_rbf, ses_values, ses_w_values, ln_values, e_values, labels, metric = 'roc', print_it=False):
    if metric == 'roc':  
        auc_se =  auroc(labels, ses_values)
        auc_se_w = auroc(labels, ses_w_values)
        auc_vnes = auroc(labels, vnes_values)
        auc_vnes_disp = auroc(labels, vnes_values_disp)
        auc_vnes_rbf = auroc(labels, vnes_values_rbf)
        auc_vnes_disp_rbf = auroc(labels, vnes_values_disp_rbf)
        auc_ln = auroc(labels, ln_values)
        auc_entr = auroc(labels, e_values)
    else: 
        auc_se =  prroc(labels, ses_values)
        auc_se_w = prroc(labels, ses_w_values)
        auc_vnes = prroc(labels, vnes_values)
        auc_vnes_disp = prroc(labels, vnes_values_disp)
        auc_vnes_rbf = prroc(labels, vnes_values_rbf)
        auc_vnes_disp_rbf = prroc(labels, vnes_values_disp_rbf)
        auc_ln = prroc(labels, ln_values)
        auc_entr = prroc(labels, e_values)
    
    if print_it: 
        print("auc_ln", auc_ln) 
        print("auc_entr", auc_entr) 
        print("auc_se",auc_se) 
        print("auc_se_w",auc_se_w)
        print("auc_vnes_disp",auc_vnes_disp)
        print("auc_vnes",auc_vnes)
    
    return auc_ln, auc_entr, auc_se, auc_se_w, auc_vnes_disp, auc_vnes, auc_vnes_disp_rbf, auc_vnes_rbf


def get_and_write_pos_performance(task_dict, list_cos, metric = "roc", model = "Mistral-7B", method = "Sampling", print_it = False): 
    vnes_values, vnes_values_disp, vnes_values_rbf, vnes_values_disp_rbf, ses_values, ses_w_values, ln_values, e_values, labels = get_position_values(list_cos)
    auc_ln, auc_entr, auc_se, auc_se_w, auc_vnes_disp, auc_vnes, auc_vnes_disp_rbf, auc_vnes_rbf = get_position_perf(vnes_values, vnes_values_disp, vnes_values_rbf, vnes_values_disp_rbf, ses_values, ses_w_values, ln_values, e_values, labels, metric = metric, print_it=print_it)

    task_dict = update_task_dict(
        task_dict, model=model, method=method,
        auc_ln_probs=auc_ln, auc_entropy=auc_entr,
        auc_se=auc_se,      auc_se_w=auc_se_w,
        auc_vne_disp_cos=auc_vnes_disp, auc_vne_contr_cos=auc_vnes,
        auc_vne_disp_rbf=auc_vnes_disp_rbf,  auc_vne_contr_rbf=auc_vnes_rbf,
    )
    return task_dict