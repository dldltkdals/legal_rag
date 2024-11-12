from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer)
import torch
import os, sys
from datasets import load_dataset, Dataset
from tqdm import tqdm
import pandas as pd
import csv
from openai import OpenAI
from retriever import Retriever, BM25_Retriever, Reranker_Retriever

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

def post_retriever():
    model="gpt-3.5-turbo"
    api_key = ""
    client = OpenAI(api_key = api_key )
    return client

class RetrievalQA:
    def __init__(self,model_name,retriever,num_docs=2,post_retrieving = False):
        self.model,self.tokenizer = load_model(model_name)
        self.retriever = retriever
        self.retriever.set_k(num_docs)
        self.post_retrieving_flag = post_retrieving
        
    def post_retrieving(self,instruction,retrieved_docs):
        client = post_retriever()
        model = "gpt-3.5-turbo"
        for doc in retrieved_docs:
            case_ = doc["context"]
            messages=[
                {"role": "system", "content": "당신은 법률 판례를 요약하는 ai입니다. 질문에 답할 수 있도록 검색된 판례를 압축해주세요."},
                {
                    "role": "user",
                    "content": f"검색된 판례들을 주어진 질문에 답을 할 수 있도록 필수적인 정보만 남겨서 두 문장으로 압축해줘.\n 질문:{instruction} \n검색된 판례:{case_}\n 압축된 판례: "
                }
            ]
            completion = client.chat.completions.create(model = model,messages=messages)
            doc["context"] = completion.choices[0].message.content
            
    def format_prompt(self,instruction):
        prompt = f"질문: \n{instruction}\n\n"
        
        prompt += "판례 요약: \n"
        retrieved_docs = self.retriever.search(instruction)
       
        if self.post_retrieving_flag == True:
            self.post_retrieving(instruction,retrieved_docs)
        print(retrieved_docs)
        for doc in retrieved_docs:
            prompt += doc["case_title"]
            prompt += ": "
            prompt += doc["context"]
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

if __name__ == "__main__":
    QA = RetrievalQA('MLP-KTLim/llama-3-Korean-Bllossom-8B',Retriever(),num_docs=2,post_retrieving = True)
    output = QA.generate("배우자가 제게 폭행을 했는데 이를 귀책사유로 이혼 할 수 있나요?")
    print(output)