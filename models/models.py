import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from huggingface_hub import notebook_login, login
import numpy as np
from huggingface_hub import InferenceClient
import os, gc

class LLM(): 
    def __init__(self, storage_type = 'local', model_id = 'openai/gpt-oss-20b'): 
        self.storage_type = storage_type
        if storage_type == 'hf_inference': 
            self.client = InferenceClient(model=model_id, token="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")
            # self.client = InferenceClient(token="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")

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
    
    def equivalence_prompt(self, text1, text2, question, mode = 'og'):
        if mode == 'og': 
            prompt = f"""We are evaluating partly evolved subsequences to the question \"{question}\"\n"""
            prompt += "Here are two possible partly evolved subsequences: \n"
            prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
            prompt += "Does Possible Answer 1 semantically entail Possible Answer 2? Respond with entailment, contradiction, or neutral."""
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
            Possible Answer 1: Kurt \n
            Possible Answer 2: The \n
            Response: No
            """
        return prompt
            

    def check_implication(self, text1, text2, question = None, mode = 'og'):
        if question is None:
            raise ValueError
        prompt = self.equivalence_prompt(text1, text2, question, mode)

        while True: 
            response = self.predict(prompt, temperature=0.02)
            if response is not None: 
                break
        
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
        
        else: 
            if 'False' in binary_response:
                return 2
            elif 'True' in binary_response:
                return 0
            else:
                return 'I am lost'
