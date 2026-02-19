import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed, pipeline
from huggingface_hub import notebook_login, login
import numpy as np
from huggingface_hub import InferenceClient
import os, gc
import re
# import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from jsonschema import validate, ValidationError

load_dotenv()


class LLM(): 
    def __init__(self, storage_type = 'local', model_id = 'openai/gpt-oss-20b'): 
        self.storage_type = storage_type
        if storage_type == 'hf_inference': 
            self.client = InferenceClient(model=model_id, api_key=os.getenv('HF_TOKEN'))
            # self.client = InferenceClient(token="")
        elif storage_type == 'open_ai_api': 
            self.client = OpenAI()
            self.model_id = model_id
        else: 
            if 'mistral' in model_id.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",           # send layers to GPU if VRAM allows
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    max_memory={
                        0: "5GiB",       # GPU 0 (your 1660 Ti) can use up to ~5 GB VRAM
                        "cpu": "16GiB"   # spill the rest to system RAM
                    }
                )
            
            if 'oss' in model_id.lower(): 
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.model = AutoModelForCausalLM.from_pretrained(
                    "openai/gpt-oss-20b",
                    device_map="auto",
                    torch_dtype="auto",
                )
                # self.model = AutoModelForCausalLM.from_pretrained(
                #     'unsloth/gpt-oss-20b-bnb-4bit',
                #     device_map="auto",
                #     load_in_4bit=True,
                #     torch_dtype="auto",
                #     trust_remote_code=True
                # )
                
                
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
    
    def predict(self, prompt, temperature = 0.9, return_all = False, response_format = None): 
        if self.storage_type == 'hf_inference': 
            message = [{"role": "user", "content": prompt}]
            out = self.client.chat_completion(
                messages = message,
                temperature=temperature,        
                top_p=0.9,
                response_format = response_format
                )
            return out.choices[0].message["content"]
        
        elif self.storage_type == "open_ai_api":
            message = [{"role": "user", "content": prompt}]
            
            # TODO: Insert response api
            # result = self.client.chat.completions.create(
            #     model=self.model_id,
            #     response_format = response_model,
            #     messages=message, 
            #     temperature=temperature
            #     )
            result = self.client.responses.create(
                model=self.model_id,
                input= message,
                text = {
                    "format" : response_format}
                #temperature=temperature
                # reasoning={
                #     "effort": "none"
                #     }
                )
            return result.output_text  
        
        else: 
            chat = [
                {"role": "user", "content": prompt}
            ]
            #inputs = self.tokenizer(prompt, return_tensors="pt")
            #inputs = {k_: v_.to(self.model.device) for k_, v_ in inputs.items()}
            
            encoded = self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            ).to(self.model.device)

            # Set pad_token_id if needed
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            with torch.no_grad():
                out = self.model.generate(
                    **encoded,  # Unpacks input_ids and attention_mask
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    return_dict_in_generate=True,
                    output_scores=True, 
                    output_logits=True, 
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=5000,
                )

            # Get prompt length
            prompt_length = encoded['input_ids'].shape[1]
            
            full_seq = out.sequences[0] 
            # print(prompt_len, len(full_seq))
            gen_ids = full_seq[prompt_length:].tolist()
            generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            if return_all: 
                return generated_text, gen_ids, out
            else: return generated_text
        
    def generate_with_topk(self, prompt: str, k: int = 25, max_new_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.9, do_sample: bool = True, seed: int | None = 42,):
        # if seed is not None:
        #     set_seed(seed)

        generated_text, gen_ids, out = self.predict(prompt=prompt, temperature=temperature, return_all=True)
        #print('generated text: ', generated_text)
        step_samples = []
        for step_idx, step_scores in enumerate(out.logits):
            # print('gen_id: ', gen_ids[step_idx])
            logits = step_scores[0]                  
            probs = torch.softmax(logits, dim=-1)     

            # # TODO Das wsl in die neue Mehtode
            # sampled_ids = torch.multinomial(probs, num_samples=k, replacement=True)
            # sampled = []
            # #print('sampled ids: ', sampled_ids)
            # current_probs = float(probs[gen_ids[step_idx]].item())

            
            # TODO Das wsl in die neue Mehtode
            # for tok_id in sampled_ids.tolist():
            #     tok_str = self.tokenizer.decode([tok_id], skip_special_tokens=False)
            #     sampled.append({
            #         "token_id": tok_id,
            #         "token_str": tok_str,
            #         "prob": float(probs[tok_id].item()),
            #     })
                # if tok_id == gen_ids[step_idx]: 
                #     current_probs = float(probs[tok_id].item())
            step_samples.append({'current_seq' : gen_ids[:step_idx+1],
                                 'prev_seq' : gen_ids[:step_idx],
                                 # 'current_seq_decoded' : self.tokenizer.decode(gen_ids[:step_idx], skip_special_tokens=True),
                                 # 'current_prob' : current_probs, 
                                 #'sampled_tokens' : sampled, 
                                 'logits' : logits.detach().cpu()})

        # gen_ids_decoded = []
        # for id in gen_ids:
        #    gen_ids_decoded.append(self.tokenizer.decode(id, skip_special_tokens=True)) 
        gen_ids_decoded = self.tokenizer.convert_ids_to_tokens(gen_ids)
        
        generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        return generated_text, step_samples, gen_ids, gen_ids_decoded
    
    # ------------------ methods for factual eval
    def validate_response(self, response, tokens, response_format): 
        validate(instance=response, schema=response_format)
        mappings = response["mappings"]

        if len(tokens) != len(mappings):
            raise ValueError("Token count mismatch")

        for i, (expected, actual) in enumerate(zip(tokens, mappings)):
            if expected != actual["token"]:
                raise ValueError(
                    f"Token mismatch at position {i}: "
                    f"expected='{expected}', got='{actual['token']}'"
                )
    
    
    def position_eval_prompt_all(self, question, answer, token_list):
        prompt = prompt = f"""You are doing a token-level semantics attribution task. Your goal is to identify which tokens carry semantic meaning for the answer.\n

            Inputs: 
            Question: {question}
            Answer: {answer}
            Tokens: {token_list}\n

            Task:
            For EACH token in tokens, classify if it directly contributed to making the answer false: \n

            "yes" = This token carries semantic value:
            - Entity names (people, places, organizations)
            - Numbers, dates, quantities, or units
            - Relationships (e.g., subject-object)
            - Negations or qualifiers 
            - Verbs or adjectives that are relevant to answering the question (if so)

            "no" = This token does not carry semantic meaning
            - Functional words, grammatically necessary but no semanitc value (e.g., "the", "is", "of")
            - Verbs and adjectives that are mostly descriptive or gramatically necessary for the sentence, but do not influence the big meaning, given the question. \n
            
            Important rules: 
            - ALL tokens MUST be classified.
            - Output length MUST exactly match the number of input tokens.
            - When uncertain, default to "no".\n
            
            \n Example Output structure: 
            {{"mappings": [
                {{"token": "The", 
                "value": "no"}},
                {{"token": "capital", 
                "value": "no"}},
                {{"token": "of", 
                "value": "no"}},
                {{"token": "France", 
                "value": "no"}},
                {{"token": "is", 
                "value": "no"}},
                {{"token": "London", 
                "value": "yes"}}]}}

            Do NOT include explanations, markdown, comments, or extra fields.
            Do NOT reorder tokens.
            Do NOT omit tokens."""
        return prompt
    
    def position_eval_prompt_bios(self, question, text, answer_false, token_list):
        prompt = prompt = f"""You are doing a token-level error attribution task. Your goal is to identify which tokens directly cause the fact to be factually incorrect within the context of the generated answer.\n

            Inputs: 
            Question: {question}
            Generated text: {text}
            False fact: {answer_false}
            Tokens: {token_list}\n

            Task:
            For EACH token in tokens, classify if it directly contributed to making the fact false: \n

            "yes" = This token is factually wrong or creates the error
            - Incorrect entity names (people, places, organizations)
            - Incorrect numbers, dates, quantities, or units
            - Incorrect relationships (e.g., subject-object swaps)
            - Negations or qualifiers that invert or invalidate a true statement
            - Verbs or adjectives that change factual meaning\n

            "no" = This token is correct or neutral
            - Grammatically necessary but not factually wrong (e.g., "the", "is", "of")
            - Factually correct in both the true and false answers
            - Contextually neutral and not responsible for the error\n
            
            Important rules: 
            - ALL tokens MUST be classified.
            - Output length MUST exactly match the number of input tokens.
            - If a multi-token entity is incorrect, mark ONLY the token(s) that are factually wrong.
            - When uncertain, default to "no".\n
            
            Output format example: 
            {{"mappings": [
                {{"token": "The", 
                "value": "no"}},
                {{"token": "capital", 
                "value": "no"}},
                {{"token": "of", 
                "value": "no"}},
                {{"token": "France", 
                "value": "no"}},
                {{"token": "is", 
                "value": "no"}},
                {{"token": "London", 
                "value": "yes"}}]}}

            Do NOT include explanations, markdown, comments, or extra fields.
            Do NOT reorder tokens.
            Do NOT omit tokens."""
        return prompt
        
    
    def position_eval_prompt_inco(self, question, answer_false, answer_true, token_list):
        prompt = prompt = f"""You are doing a token-level error attribution task. Your goal is to identify which tokens directly cause the answer to be factually incorrect.\n

            Inputs: 
            Question: {question}
            True answer: {answer_true}
            False Answer: {answer_false}
            Tokens: {token_list}\n

            Task:
            For EACH token in tokens, classify if it directly contributed to making the answer false: \n

            "yes" = This token is factually wrong or creates the error
            - Incorrect entity names (people, places, organizations)
            - Incorrect numbers, dates, quantities, or units
            - Incorrect relationships (e.g., subject-object swaps)
            - Negations or qualifiers that invert or invalidate a true statement
            - Verbs or adjectives that change factual meaning\n

            "no" = This token is correct or neutral
            - Grammatically necessary but not factually wrong (e.g., "the", "is", "of")
            - Factually correct in both the true and false answers
            - Contextually neutral and not responsible for the error\n
            
            Important rules: 
            - ALL tokens MUST be classified.
            - Output length MUST exactly match the number of input tokens.
            - If a multi-token entity is incorrect, mark ONLY the token(s) that are factually wrong.
            - When uncertain, default to "no".\n
            
            Output format example: 
            {{"mappings": [
                {{"token": "The", 
                "value": "no"}},
                {{"token": "capital", 
                "value": "no"}},
                {{"token": "of", 
                "value": "no"}},
                {{"token": "France", 
                "value": "no"}},
                {{"token": "is", 
                "value": "no"}},
                {{"token": "London", 
                "value": "yes"}}]}}

            Do NOT include explanations, markdown, comments, or extra fields.
            Do NOT reorder tokens.
            Do NOT omit tokens."""
        return prompt
    
    def check_positions(self, question, answer_false, answer_true, token_list, generated_text = None, mode = 'detect_inco'):
        # class TokenMapping(BaseModel):
        #     mappings: dict[str, str] = Field(default_factory=dict)
        
        # response_model = TokenMapping()
        if self.storage_type == 'hf_inference':
            response_format = {
                "type": "json_schema",
                "value" : {
                    "properties": {
                        "mappings": { "type": "array", "items" : {"type" : "json_schema", "value" : {"properties" : {"token" : {"type" : "string"}, "value": {"type" : "string"}}}}}
                        }
                    },
                "required": ["mappings"]
                }
        else:
            response_format = {
                "name": "response_mappings",
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "mappings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "token": { "type": "string" },
                                    "value": { "type": "string" }
                                },
                                "required": ["token", "value"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["mappings"],
                    "additionalProperties": False
                }
            }                                
                            

             
        if mode == 'detect_inco': 
            if generated_text is not None: 
                prompt = self.position_eval_prompt_inco(question, answer_false, answer_true, token_list)
            else:
                prompt = self.position_eval_prompt_bios(question, generated_text, answer_false, token_list)
        else: 
            prompt = self.position_eval_prompt_all(question, answer_false, token_list)
        result = self.predict(prompt = prompt, temperature = 0.01, response_format = response_format)
        #self.validate_response(result, token_list, response_format)
        # assert len(result.mappings) == len(token_list)
        return result
    
    # ------------------- methods for NLI
    def equivalence_prompt(self, text1, text2, question, mode = 'og'):
        if mode == 'og': 
            prompt = f"""We are evaluating partly evolved subsequences to the question \"{question}\"\n"""
            prompt += "Here are two possible partly evolved subsequences: \n"
            prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
            prompt += "Does Possible Answer 1 semantically entail Possible Answer 2? Respond with entailment, contradiction, or neutral."""
        else: 
            # prompt = f"""We are evaluating partly evolved subsequences. 
            # Will subsequence 1 semantically lead to a completley different meaning to the question than subsequence 2? Completley different means that, regradless of what tokens are added in the future, both subsequences already contradict each other at this state..
            # Will subsequence 1 semantically lead to the same meaning to the question as subsequence 2? The same means that the most important aspects about the answer are already within the subsequences and regradless of what tokens are added in the future, this meaning will not change.
            # Respond with contradiction if the subsequences will surely lead to different meanings, with entailment if they will lead to the same meaning and with neutral if you are unsure or one can not tell yet. If the subsequences are at the very start and you are not sure, also respond with neutral.       
            
            # Here are some examples: \n
            # Example 1: \n
            # Question: Who was the lead singer of Nirvana? \n
            # Subsequence 1: The lead singer was Kurt \n
            # Subsequence 2: The lead singer was Tom  \n
            # Response: contradiction\n
            
            # Example 2: \n
            # Question: Who was the lead singer of Nirvana? \n
            # Subsequence 1: The \n
            # Subsequence 2: Kurt \n
            # Response: neutral\n
            
            # Subsequences to be analyzed: \n 
            # Question:{question}\n 
            # Subsequence 1: {text1}\nSubsequence 2: {text2}\n     
            # Response: <One word only>
            
            # FINAL INSTRUCTIONS — READ CAREFULLY:
            # You MUST answer using exactly ONE word.
            # You MUST choose one of: contradiction, neutral, entailment.
            # Do NOT explain your answer.
            # Do NOT show your reasoning.
            # Do NOT output analysis or internal thoughts.
            # If you are unsure, answer: neutral.

            # FINAL ANSWER (one word only):
            # """
            
            prompt1 =f"""Compare two partial text subsequences and determine their semantic relationship
                    ("contradiction", "entailment", or "neutral") based ONLY on their final token.
                    
                    Follow this reasoning: 
                    1. For each subsequence, determine if the last token is semantically relevant.
                    2. If both final tokens are semantically irrelevant, return "entailment"
                    3. If one final token is semantically relevant and the other one is irrelevant, return "neutral"
                    4. If both final token are semantically relevant, return:  
                        - "contradiction": subsequences will inevitably lead to different meanings, regardless of future tokens. The last tokens make them mutually exclusive.
                        (e.g. different entity names, different titles, different dates, different names, different professions etc.)
                        - "entailment": The subsequences will lead to the same meaning, regardless of future tokens. The last tokens are semantically equivalent (e.g., synonyms).
                        - "neutral": Cannot determine yet whether they'll diverge or converge. The last token doesn't provide enough information, or future tokens could make them equivalent despite current differences.
                        (e.g. functional tokens, different topics the token refer to)
                        
                    Examples: 
                    Question: Who was the lead singer of Nirvana?
                    Subsequence 1: The lead singer was Kurt
                    Subsequence 2: The lead singer was Tom
                    Answer: contradiction
                    (Both semantically relevant, different singer names --> contradiction)

                    Question: Who was the lead singer of Nirvana?
                    Subsequence 1: The lead singer of Nirvana, who was born
                    Subsequence 2: The lead singer of Nirvana, who was known
                    Answer: neutral
                    (Both semantically relevant, but refer to different topics (birth vs. fame) --> neutral)

                    Question: What color is the sky?
                    Subsequence 1: The sky is blue
                    Subsequence 2: The sky appears blue
                    Answer: entailment
                    (Both semantically irrelevant --> entailment)

                    Question: What is the title of J.R.R. Tolkien's most famous series?
                    Subsequence 1: The title is "
                    Subsequence 2: The title is Lord
                    Answer: neutral
                    (Both semantically relevant, " just another stylistiv way of expressing a title, no fixed value yet --> neutral)

                    Question: When did Madonna graduate?
                    Subsequence 1: Madonna graduated in New York
                    Subsequence 2: Madonna graduated in painting
                    Answer: neutral
                    (Both semantically relevant, but refer to differen topics (place vs. study subject) --> neutral)

                    NOW ANALYZE:

                    Question: {question}
                    Subsequence 1: {text1}
                    Subsequence 2: {text2}

                    Answer:"""
                             
            
            prompt = f"""Task:
                    Compare two partial text subsequences and determine their semantic relationship
                    ("contradiction", "entailment", or "neutral") based ONLY on their final token.

                    DEFINITIONS:
                    - Atomic fact: An independent piece of information answering a specific aspect of the question (e.g., identity, time, place, property, event).
                    - A final token refers to a atomic fact if it adds information to an already existing atmomic fact or if it is the beginning of a new, not yet covered topic so e.g. a new property, relation, event, additonal detail etc.

                    REASONING PATH (MUST be followed):
                    1. For EACH subsequence determine the atomic fact that the final token refers to or introduces.
                    2. If each final token refers to a different atomic fact --> return "neutral"
                    3. If both final tokens refer to the same atomic fact:
                        - "contradiction": subsequences will inevitably lead to different meanings, regardless of future tokens. The last tokens make them mutually exclusive.
                        (e.g. different entity names, different titles, different dates, different names, different professions etc.)
                        - "entailment": The subsequences will lead to the same meaning, regardless of future tokens. The last tokens are semantically equivalent (e.g., synonyms).
                        - "neutral": Cannot determine yet whether they'll diverge or converge. The last token doesn't provide enough information, or future tokens could make them equivalent despite current differences.
                        (e.g. functional tokens, ordering differences within the fact itself)
                    
                    EXAMPLES:

                    Question: Who was the lead singer of Nirvana?
                    Subsequence 1: The lead singer was Kurt
                    Subsequence 2: The lead singer was Tom
                    Answer: contradiction
                    (Same atomic fact: lead singer identity. Incompatible values --> contradiction)

                    Question: Who was the lead singer of Nirvana?
                    Subsequence 1: The lead singer of Nirvana, who was born
                    Subsequence 2: The lead singer of Nirvana, who was known
                    Answer: neutral
                    (Different atomic facts: birth vs fame --> neutral)

                    Question: What color is the sky?
                    Subsequence 1: The sky is blue
                    Subsequence 2: The sky appears blue
                    Answer: entailment
                    (Same atomic fact, only stylistic difference in word choice --> entailment)

                    Question: What is the title of J.R.R. Tolkien's most famous series?
                    Subsequence 1: The title is "
                    Subsequence 2: The title is Lord
                    Answer: neutral
                    (Same atomic fact, " just another stylistiv way of expressing a title, no fixed value yet --> neutral)

                    Question: When did Madonna graduate?
                    Subsequence 1: Madonna graduated in New York
                    Subsequence 2: Madonna graduated in painting
                    Answer: neutral
                    (Different atomic facts: place vs field --> neutral)

                    NOW ANALYZE:

                    Question: {question}
                    Subsequence 1: {text1}
                    Subsequence 2: {text2}

                    Answer:"""
        # elif mode == 'adapted': 
        #    prompt = f"""We are evaluating partly evolved subsequences to the question \"{question}\"\n Here are two possible partly evolved subsequences: \n
        #     Sequence 1: {text1}\nSequence 2: {text2}\n
        #     Will Sequence 1 semantically lead to a completley different meaning to the question than Sequence 2? 
        #     Completley different means that there is no way that both sequences can have the same semantic meaning when more tokens are added in the future.
        #     Respond with False if the sequences will surely lead to different answers and with True if not.
            
        #     Example 1: \n
        #     Question: Who was the lead singer of Nirvana? \n
        #     Possible Answer 1: The lead singer was Kurt \n
        #     Possible Answer 2: The lead singer was Tom  \n
        #     Response: False
            
        #     Example 1: \n
        #     Question: Who was the lead singer of Nirvana? \n
        #     Possible Answer 1: The lead singer was Kurt \n
        #     Possible Answer 2: The lead singer was the \n
        #     Response: True
        #     """
            # prompt = f"""Task: Determine if two partial text subsequences will inevitably contradict each other based on their content so far.

            #     DEFINITIONS:
            #     - "contradiction": The subsequences have already diverged into mutually exclusive meanings that cannot be reconciled by future tokens (e.g., different factual claims about the same aspect)
            #     - "neutral": The subsequences either could still converge to the same meaning, address different aspects that don't contradict, or differ only stylistically

            #     EXAMPLES:
            #     Question: Who was the lead singer of Nirvana?
            #     Subsequence 1: The lead singer was Kurt
            #     Subsequence 2: The lead singer was Tom
            #     Answer: contradiction
            #     Reason: Different names for the same role = incompatible factual claims

            #     Question: Who was the lead singer of Nirvana?
            #     Subsequence 1: The
            #     Subsequence 2: Kurt
            #     Answer: neutral
            #     Reason: "The" could continue as "The lead singer Kurt..." - too early to determine

            #     NOW ANALYZE:
            #     Question: {question}
            #     Subsequence 1: {text1}
            #     Subsequence 2: {text2}

            #     OUTPUT INSTRUCTIONS:
            #     - Default to "neutral" when uncertain
            #     - Only compare the two sequences, do not try to answer the question they refer to!
            #     - Respond with EXACTLY ONE WORD: "contradiction" or "neutral"

            #     Answer:"""
             
        return prompt
            

    def check_implication(self, text1, text2, question = None, mode = 'og'):
        if question is None:
            raise ValueError
        prompt = self.equivalence_prompt(text1, text2, question, mode)
        counter = 0
        while counter < 3: 
            response = self.predict(prompt, temperature=0.02)
            counter = counter + 1
            if response is not None: 
                break
        
        if response is not None: 
            binary_response = response.lower()[:30]
            
            if mode == 'og':
                if 'entailment' in binary_response:
                    return 2
                elif 'neutral' in binary_response:
                    return 1
                elif 'contradiction' in binary_response:
                    return 0
                else:
                    return -1
            elif mode == 'data': 
                if '-1' in binary_response:
                    return -1
                elif '1' in binary_response:
                    return 1
                elif '0' in binary_response:
                    return 0
                else:
                    return -100000
            elif mode == 'adapted': 
                if 'True' in binary_response:
                    return 2
                elif 'False' in binary_response:
                    return 0
                else:
                    return -1
        else: 
            if mode == 'data': 
                return -100000

    def check_implication_batch(self, batch_pairs, question, mode="data"):
        prompts = [
            self.equivalence_prompt(t1, t2, question=question, mode=mode)
            for (t1, t2) in batch_pairs
        ]
        
        chats = [
            [{"role": "user", "content": prompt}] for prompt in prompts            
        ]
        
        
        if self.storage_type == 'local':
            # Apply the template to each batch element
            batched_chats = [
                self.tokenizer.apply_chat_template(
                    chat,
                    add_generation_prompt=True,
                    tokenize=False
                )
                for chat in chats
            ]
            
            outputs = self.pipe(
                batched_chats,
                max_new_tokens=5000,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text = False,
                clean_up_tokenization_spaces = True
            )
        elif self.storage_type == "open_ai_api": 
            chats = [
                [{"role": "user", "content": prompt}] for prompt in prompts            
            ]

            # Process each chat (OpenAI API doesn't support batching in a single call)
            outputs = []
            for chat in chats:
                # response = self.client.chat.completions.create(
                #     model=self.model_id,  # or "gpt-3.5-turbo", "gpt-4-turbo", etc.
                #     messages=chat,
                #     max_tokens=5000,
                #     temperature=0,  # equivalent to do_sample=False
                # )
                # outputs.append(response.choices[0].message.content)
                result = self.client.responses.create(
                    model=self.model_id,
                    input= chat
                    #temperature=0
                # reasoning={
                #     "effort": "none"
                #     }
                )
                outputs.append(result.output_text)
            
        
        def extract_label(text, mode):
            if mode == "data": 
                matches = re.findall(r'\b(entailment|contradiction|neutral)\b', text, flags=re.IGNORECASE)
            elif mode == "adapted": 
                # matches = re.findall(r'\b(neutral|contradiction)\b', text, flags=re.IGNORECASE)
                matches = re.findall(r'\b(entailment|contradiction|neutral)\b', text, flags=re.IGNORECASE)
            if not matches:
                return None
            return matches[-1].lower()
        
        scores = []
        for i, out in enumerate(outputs):
            if self.storage_type == 'local':
                full = out[0]["generated_text"]
            elif self.storage_type == 'open_ai_api': 
                full = out
            label = extract_label(full, mode)
            if mode == 'data':
                mapping = {
                    "contradiction": -1,
                    "neutral": 0,
                    "entailment": 1,
                }
                value = mapping.get(label, -100000)
            elif mode == 'og':
                mapping = {
                    "contradiction": 0,
                    "neutral": 1,
                    "entailment": 2,
                }
                value = mapping.get(label, -1)
            elif mode == 'adapted': 
                mapping = {
                    "contradiction": 0,
                    "neutral": 1,
                    "entailment": 2}                
                value = mapping.get(label, 1)
            
            scores.append(value)
        return scores
    
    