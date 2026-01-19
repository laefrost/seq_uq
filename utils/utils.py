import wandb
import pickle
import os
import logging
import argparse

def save(file, name):
    #with open(f'{wandb.run.dir}/{file}', 'wb') as f:
    with open(name, 'wb') as f:
        pickle.dump(file, f)
    #wandb.save(f'{wandb.run.dir}/{file}')
    
def load(name):
    with open(name, 'rb') as file:
        a = pickle.load(file)
    return a

def setup_logger():
    """Setup logger to always print time and level."""
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)

def get_parser():
    parser = argparse.ArgumentParser()       
    parser.add_argument('--random_seed', type=int, default=10)
    parser.add_argument(
        "--metric", type=str, default="llm",
        choices=['llm'],
        help="Metric to assign accuracy to generations.")

    parser.add_argument(
        "--model_id", type=str, default="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ", help="Model ID",
    )
    parser.add_argument(
        "--emb_model_id", type=str, default="all-MiniLM-L6-v2", help="Embedding Model ID"
        # choices=['all-MiniLM-L6-v2', 
        #          'intfloat/e5-large-v2', 
        #          'models_peft/all-MiniLM-L6-v2-peft/final',
        #          'models_peft_og/all-MiniLM-L6-v2-peft/final']
    )
    
    parser.add_argument(
        "--emb_model_id_deltas", type=str, default="all-MiniLM-L6-v2", help="Embedding Model ID"
        # choices=['all-MiniLM-L6-v2', 
        #          'intfloat/e5-large-v2', 
        #          'models_peft/all-MiniLM-L6-v2-peft/final',
        #          'models_peft_og/all-MiniLM-L6-v2-peft/final']
    )
    
    
    parser.add_argument(
        "--eval_model_id", type=str, default="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ", help="Model ID of evaluation model",
    )
    parser.add_argument(
        "--ellm_model_id", type=str, default="openai/gpt-oss-20b", help="Model ID of NLI model",
    )
    parser.add_argument(
        "--fact_model_name", type=str, default="ChatGPT", help="Model name of fact score model", choices=["retrieval+llama", "retrieval+llama+npm", "retrieval+ChatGPT", "npm", 
                              "retrieval+ChatGPT+npm", "ChatGPT", "gpt-oss", "retrieval+gpt-oss-20b", 
                              "hf-inf", "retrieval+hf-inf"],
    )
    
    parser.add_argument(
        "--consider_types", default=False,
        action=argparse.BooleanOptionalAction,
        help='Mask word types during UQ'
    )
    
    parser.add_argument(
        "--model_max_new_tokens", type=int, default=50,
        help="Max number of tokens generated.",
    )
    parser.add_argument(
        "--dataset", type=str, default="trivia_qa",
        choices=['trivia_qa', 'factual_bio', 'trivia_qa_data'],
        help="Dataset to use")
    
    parser.add_argument(
        "--exp_name", type=str, help="Experiment name used for saving")
        
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="Number of samples to use")
       
    parser.add_argument(
        "--k", type=int, default=10,
        help="Number of generations to use")
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Temperature")
    
    parser.add_argument(
        "--task_type", default='qa', type=str)
    
    parser.add_argument(
        "--eval_type", default='fact_score', type=str)
    
    parser.add_argument(
        "--compute_uncertainties", default=False,
        action=argparse.BooleanOptionalAction,
        help='Trigger compute_uncertainty_measures.py')
    
    parser.add_argument(
        "--eval_answers", default=False,
        action=argparse.BooleanOptionalAction,
        help='Trigger eval_answers.py')
    return parser


def construct_prompt(question, task_type = 'qa'):
    if task_type == 'qa': 
        return f"You are a helpful assistant. Answer the following question in one single, short sentence. Only output this sentence. {question}" 
    elif task_type == 'bio': 
        return f"You are a helpful assistant. {question}" 
    

def model_based_metric(predicted_answer, example, model):
    if 'answers' in example:
        correct_answers = example['answers']['text']
    elif 'reference' in example:
        correct_answers = example['reference']['answers']['text']
    elif 'answer' in example:
        correct_answers = example['answer']['aliases']
    else:
        raise ValueError

    prompt = f'We are assessing the quality of answers to the following question: {example["question"]}\n'
    if len(correct_answers) == 1:
        prompt += f"The expected answer is: {correct_answers[0]}.\n"
    else:
        prompt += f"The following are expected answers to this question: {correct_answers}.\n"

    prompt += f"The proposed answer is: {predicted_answer}\n"

    if len(correct_answers) == 1:
        prompt += "Within the context of the question, does the proposed answer mean the same as the expected answer?"
    else:
        prompt += "Within the context of the question, does the proposed answer mean the same as any of the expected answers?"

    prompt += " Respond only with yes or no.\nResponse:"

    
    # # TODO: adapt that 
    # if 'gpt' in model.model_name.lower():
    #     predicted_answer = model.predict(prompt, 0.01)
    # else:
    #     predicted_answer, _, _ = model.predict(prompt, 0.01)
    
    # predicted_answer = None
    # while predicted_answer == None: 
    #    print('while')
    predicted_answer = model.predict(prompt, temperature = 0.01)
    
    print(predicted_answer)

    if 'yes' in predicted_answer.lower():
        return 1.0
    elif 'no' in predicted_answer.lower():
        return 0.0
    else:
        logging.warning('Redo llm check.')
        predicted_answer = model.predict(prompt, 1)
        if 'yes' in predicted_answer.lower():
            return 1.0
        elif 'no' in predicted_answer.lower():
            return 0.0

        logging.warning('Answer neither no nor yes. Defaulting to no!')

def llm_metric(predicted_answer, example, model):
    return model_based_metric(predicted_answer, example, model)    
    
def get_metric(metric = 'llm'):
    # Reuses the globally active model for these.
    if metric == 'llm':
        metric = llm_metric
    # elif metric == 'llm_gpt-3.5':
    #     metric = get_gpt_metric(metric)
    # elif metric == 'llm_gpt-4':
    #     metric = get_gpt_metric(metric)
    else:
        raise ValueError

    return metric
