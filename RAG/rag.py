from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer)
import torch
from rerank.reranker import Reranker

from tqdm import tqdm
from openai import OpenAI
from retriever import Retriever, BM25_Retriever

SYS_PROMPT = '''You are a helpful AI assistant for legal tips. Please answer the user's questions kindly with their questions on cases.
    당신은 유능한 법률 AI 어시스턴트 입니다.사용자의 질문에 답하려면 질문과 함께 제공되는 판례를 참고 하여 답해주세요.
    
    아래의 규칙들을 반드시 지켜서 대답을 생성해주세요.
    
    1.최대한 간결한 문장으로 대답해주세요.
    2.대답을 지어내려하지 마세요.
    3.입력과 함께 제공되는 판례를 통해 대답을 생성할 수 없는 경우 "해당 질문에 대한 판례 유사한 판례를 찾을 수 없습니다. 정확한 상담을 위해 법률 전문가와 상의하세요."라는 답변을 생성하세요.
    '''
    
def load_model(base_model):
    torch_dtype = torch.float16
    attn_implementation = "eager"

    # 모델,토크나이저 준비
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
            )
    return model, tokenizer


class RetrievalQA:
    def __init__(self,model_name,retriever,top_k=3):
        self.model,self.tokenizer = load_model(model_name)
        self.retriever = retriever
        self.top_k = top_k
            
    def format_prompt(self,instruction):
        prompt = f"질문: \n{instruction}\n\n"
        
        prompt += "판례 요약: \n"
        if isinstance(self.retriever, Reranker):
            retrieved_docs = self.retriever.search(instruction)
        else:
            retrieved_docs = self.retriever.search(instruction,self.top_k)
            
       
    
        for doc in retrieved_docs:
            prompt += doc.metadata["case_title"]
            prompt += ": "
            prompt += doc.metadata["summary"]
            prompt += "\n\n"
        return prompt
    
    def get_prompt(self, instruction):
        
        messages = [
            {"role": "system", "content": f"{SYS_PROMPT}"},
            {"role": "user", "content": f"{self.format_prompt(instruction)}"}
            ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, 
                                        add_generation_prompt=True)

    def generate(self,instruction):
        prompt = self.get_prompt(instruction)
        
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, 
                    truncation=True).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=512,
                eos_token_id = self.tokenizer.eos_token_id,
                penalty_alpha=0.31,
                top_k=4,
                repetition_penalty = 1.02,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant\n\n")[-1]
