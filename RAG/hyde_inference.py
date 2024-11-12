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
from argparse import ArgumentParser
from rag import RetrievalQA
from retriever import Retriever, BM25_Retriever, Reranker_Retriever
import hyde
from hyde.generator import OpenAIGenerator
from hyde.promptor import Promptor
from hyde.hyde import HyDE

SYS_PROMPT = '''You are a helpful AI assistant for legal tips. Please answer the user's questions kindly with their questions on cases.
    당신은 유능한 법률 AI 어시스턴트 입니다.사용자의 질문에 답하려면 질문과 함께 제공되는 판례를 참고 하여 답해주세요.
    
    아래의 규칙들을 반드시 지켜서 대답을 생성해주세요.
    
    1.최대한 간결한 문장으로 대답해주세요.
    2.대답을 지어내려하지 마세요.
    3.입력과 함께 제공되는 판례를 통해 대답을 생성할 수 없는 경우 "해당 질문에 대한 판례 유사한 판례를 찾을 수 없습니다. 정확한 상담을 위해 법률 전문가와 상의하세요."라는 답변을 생성하세요.
    '''
def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data', type=str, default="../data/test_set.csv")
    parser.add_argument('--model_name', type=str, default='MLP-KTLim/llama-3-Korean-Bllossom-8B')
    parser.add_argument('--output', type=str, default="output")
    parser.add_argument('--top_k', type=int, help = "num_docs to search",default= 2)
   
    
    args = parser.parse_args()
    return args.__dict__

def load_data(data_path):
    df = pd.read_csv(data_path)
    data = df.to_dict(orient='records')
    dataset = Dataset.from_list(data)
    return dataset

def prompt_format(prompt,tokenizer):
    
    messages = [
        {"role": "system", "content": f"{SYS_PROMPT}"},
        {"role": "user", "content": f"{prompt}"}
        ]
    return tokenizer.apply_chat_template(messages, tokenize=False, 
                                   add_generation_prompt=True)
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
def result_format(documents):
        result = []
        for doc in documents:
            result.append({
                "question":doc.page_content, 
                "case_title": doc.metadata["case_title"],
                "context":doc.metadata["summary"],
                })
        return result
def get_prompt(instruction,docs):
    prompt = f"질문: \n{instruction}\n\n"
    for doc in docs:
        prompt += doc["case_title"]
        prompt += ": "
        prompt += doc["context"]
        prompt += "\n\n"
    return prompt
def hyde_inference(config,hyde):
    data_path,output = config["data"],config["output"]+ ".csv"
    top_k = config["top_k"]
    #데이터셋 준비
    
    dataset = load_data(data_path)
    model, tokenizer = load_model('MLP-KTLim/llama-3-Korean-Bllossom-8B')
    with open(os.path.join("output",output), mode = 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(["question","answer","generated_answer"])
        for row in tqdm(dataset,desc = "inferencing..."):
            instruction = row["question"]
            answer = row["answer"]
            docs = hyde_.e2e_search(instruction, k = top_k)
            docs = result_format(docs)
            print("top_k: ", len(docs))
            
            prompt = get_prompt(instruction,docs)
            messages = [
                        {"role": "system", "content": f"{SYS_PROMPT}"},
                        {"role": "user", "content": f"{prompt}"}
                        ]
            inputs = tokenizer.apply_chat_template(messages, tokenize=False, 
                                   add_generation_prompt=True)
            
            inputs = tokenizer(inputs, return_tensors='pt', padding=True, 
                    truncation=True).to(model.device)
            outputs = model.generate(
                            **inputs,
                            do_sample=True,
                            max_new_tokens=512,
                                eos_token_id = tokenizer.eos_token_id,
                                penalty_alpha=0.31,
                                top_k=4,
                                repetition_penalty = 1.02,
                        )
            output = tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant\n\n")[-1]
     
            writer.writerow([instruction,answer,output])



if __name__ == "__main__":
    config = parse_args()
    retriever = Retriever()
    searcher = retriever.retriever
    encoder = retriever.model
    model_name="gpt-3.5-turbo"
    api_key = ""
    generator = OpenAIGenerator(model_name=model_name, api_key=api_key)
    promptor = Promptor()
    hyde_ = HyDE(promptor, generator,encoder,searcher)
    if not os.path.exists("output"):
        print("create new folder")
        os.mkdir("output")
    hyde_inference(config,hyde=hyde_)
   
    

    