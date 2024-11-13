import os
from datasets import  Dataset
from tqdm import tqdm
import pandas as pd
import csv
from argparse import ArgumentParser


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





    