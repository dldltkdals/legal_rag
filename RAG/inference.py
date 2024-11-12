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

def inference(QA,config):
    data_path,output = config["data"],config["output"]+ ".csv"
    #데이터셋 준비
    dataset = load_data(data_path)

    with open(os.path.join("output",output), mode = 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(["question","answer","generated_answer"])
        for row in tqdm(dataset,desc = "inferencing..."):
            instruction = row["question"]
            answer = row["answer"]
            output = QA.generate(instruction)
            writer.writerow([instruction,answer,output])



if __name__ == "__main__":
    config = parse_args()
    QA = RetrievalQA(model_name=config["model_name"],retriever=Reranker_Retriever(),num_docs=config["top_k"])
    if not os.path.exists("output"):
        print("create new folder")
        os.mkdir("output")
    inference(QA,config)

    