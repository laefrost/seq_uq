import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed, pipeline
from huggingface_hub import notebook_login, login
import numpy as np
from huggingface_hub import InferenceClient
import os, gc

class LLM(): 
    def __init__(self, storage_type = 'local', model_id = 'openai/gpt-oss-20b'): 
        self.storage_type = storage_type
        if storage_type == 'hf_inference': 
            self.client = InferenceClient(model=model_id, token="")
            # self.client = InferenceClient(token="")

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
                # self.model = AutoModelForCausalLM.from_pretrained(
                #     "openai/gpt-oss-20b",
                #     device_map="auto",
                #     torch_dtype=torch.float16,
                # )
                self.model = AutoModelForCausalLM.from_pretrained(
                    'unsloth/gpt-oss-20b-bnb-4bit',
                    device_map="auto",
                    load_in_4bit=True,
                    torch_dtype="auto",
                    trust_remote_code=True
                )
    
    def predict(self, prompt, temperature = 0.9, return_all = False): 
        if self.storage_type == 'hf_inference': 
            message = [{"role": "user", "content": prompt}]
            
            
            out = self.client.chat_completion(
                messages = message,
                temperature=temperature,        
                top_p=0.9,                 
            )

            return out.choices[0].message["content"]
        else: 
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k_: v_.to(self.model.device) for k_, v_ in inputs.items()}
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    return_dict_in_generate=True,
                    output_scores=True,  
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=500,
                )

            prompt_len = inputs["input_ids"].shape[-1]
            
            full_seq = out.sequences[0] 
            # print(prompt_len, len(full_seq))
            gen_ids = full_seq[prompt_len:].tolist()
            generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            if return_all: 
                return generated_text, gen_ids, out
            else: return generated_text
        
    def generate_with_topk(self, prompt: str, k: int = 25, max_new_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.9, do_sample: bool = True, seed: int | None = 42,):
        if seed is not None:
            set_seed(seed)

        generated_text, gen_ids, out = self.predict(prompt=prompt, temperature=temperature, return_all=True)
        #print('generated text: ', generated_text)
        step_samples = []
        for step_idx, step_scores in enumerate(out.scores):
            # print('gen_id: ', gen_ids[step_idx])
            logits = step_scores[0]                  
            probs = torch.softmax(logits, dim=-1)     

            sampled_ids = torch.multinomial(probs, num_samples=k, replacement=True)
            sampled = []
            #print('sampled ids: ', sampled_ids)
            current_probs = float(probs[gen_ids[step_idx]].item())

            for tok_id in sampled_ids.tolist():
                tok_str = self.tokenizer.decode([tok_id], skip_special_tokens=False)
                sampled.append({
                    "token_id": tok_id,
                    "token_str": tok_str,
                    "prob": float(probs[tok_id].item()),
                })
                # if tok_id == gen_ids[step_idx]: 
                #     current_probs = float(probs[tok_id].item())
            step_samples.append({'current_seq' : gen_ids[:step_idx],
                                 # 'current_seq_decoded' : self.tokenizer.decode(gen_ids[:step_idx], skip_special_tokens=True),
                                 'current_prob' : current_probs, 
                                 'sampled_tokens' : sampled})

        generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        return generated_text, step_samples, gen_ids
    
class EntailmentLLM(LLM):
    def __init__(self, storage_type='local', model_id=None):
        super().__init__(storage_type, model_id)
        if storage_type == "local":
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            self.pipe = None
    
    def equivalence_prompt(self, text1, text2, question, mode = 'og'):
        if mode == 'og': 
            prompt = f"""We are evaluating partly evolved subsequences to the question \"{question}\"\n"""
            prompt += "Here are two possible partly evolved subsequences: \n"
            prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
            prompt += "Does Possible Answer 1 semantically entail Possible Answer 2? Respond with entailment, contradiction, or neutral."""
        elif mode == 'data': 
            prompt = f"""We are evaluating partly evolved subsequences to the question \"{question}\"\n Here are two possible partly evolved subsequences: \n
            Sequence 1: {text1}\nSequence 2: {text2}\n
            Will Sequence 1 semantically lead to a completley different meaning to the question than Sequence 2? Completley different means that there is no way that both sequences can have the same semantic meaning when more tokens are added in the future.
            Will Sequence 1 semantically lead to the same meaning to the question as Sequence 2? The same means that the most important aspects about the answer are already within the sequence and regradless of the extra fluff we add, this meaning will not change.
            Respond with -1 if the sequences will surely lead to different meanings, with 1 if the will lead to the same meaning and with 0 if ypu are unsure or one can not tell yet.          
            Example 1: \n
            Question: Who was the lead singer of Nirvana? \n
            Possible Answer 1: The lead singer was Kurt \n
            Possible Answer 2: The lead singer was Tom  \n
            Response: -1
            
            Example 2: \n
            Question: Who was the lead singer of Nirvana? \n
            Possible Answer 1: The lead singer was Kurt \n
            Possible Answer 2: The lead singer was the \n
            Response: 0
            
            Example 3: \n
            Question: In what did Madonna graduate? \n
            Possible Answer 1: Madonna graduated in arts \n
            Possible Answer 2: Madonna graduated in painting \n
            Response: 1
            
            Example 4: \n
            Question: In what did Madonna graduate? \n
            Possible Answer 1: Madonna graduated in swimming \n
            Possible Answer 2: Madonna graduated in painting \n
            Response: -1
            
            Example 5: \n
            Question: In what did Madonna graduate? \n
            Possible Answer 1: Madonna graduated in 1998 \n
            Possible Answer 2: Madonna graduated in painting \n
            Response: 0
            """
        else: 
           prompt = f"""We are evaluating partly evolved subsequences to the question \"{question}\"\n Here are two possible partly evolved subsequences: \n
            Sequence 1: {text1}\nSequence 2: {text2}\n
            Will Sequence 1 semantically lead to a completley different meaning to the question than Sequence 2? Completley different means that there is no way that both sequences can have the same semantic meaning when more tokens are added in the future.
            Respond with True if the sequences will surely lead to different answers and with False if not.
            
            Example 1: \n
            Question: Who was the lead singer of Nirvana? \n
            Possible Answer 1: The lead singer was Kurt \n
            Possible Answer 2: The lead singer was Tom  \n
            Response: Yes
            
            Example 1: \n
            Question: Who was the lead singer of Nirvana? \n
            Possible Answer 1: The lead singer was Kurt \n
            Possible Answer 2: The lead singer was The \n
            Response: No
            """
             
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
                    return 'I am lost'
            elif mode == 'data': 
                if '-1' in binary_response:
                    return -1
                elif '1' in binary_response:
                    return 1
                elif '0' in binary_response:
                    return 0
                else:
                    return -100000
            else: 
                if 'False' in binary_response:
                    return 2
                elif 'True' in binary_response:
                    return 0
                else:
                    return 'I am lost'
        else: 
            if mode == 'data': 
                return -100000

    def check_implication_batch(self, batch_pairs, question, mode="data"):
        prompts = [
            self.equivalence_prompt(t1, t2, question=question, mode=mode)
            for (t1, t2) in batch_pairs
        ]

        if self.storage_type == "hf_inference":
            return [self.check_implication(t1, t2, question, mode) for t1, t2 in batch_pairs]

        outputs = self.pipe(
            prompts,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        scores = []
        for out in outputs:
            if out[0]["generated_text"] is not None: 
                text = out[0]["generated_text"].lower()

                if mode == 'data':
                    if '-1' in text:
                        scores.append(-1)
                    elif '1' in text:
                        scores.append(1)
                    elif '0' in text:
                        scores.append(0)
                    else:
                        scores.append(-100000)
                else:
                    scores.append(text)
            else: 
                scores.append(-100000)

        return scores