from nltk.translate.bleu_score import corpus_bleu
from transformers import AutoTokenizer

from korouge_score import rouge_scorer
from bert_score import score
import pandas as pd
import os


base_model = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'
tokenizer = AutoTokenizer.from_pretrained(base_model)
def load_csv(path):
    df = pd.read_csv(path)
    ref, cand = [],[]
    
    for idx, row in df.iterrows():
        cand.append(row["generated_answer"])
        ref.append(row["answer"])
    return ref,cand
def get_bert_score(paht):
 
    ref,cand = load_csv(path)
    P, R, F1 = score(cand, ref, model_type = "jhgan/ko-sroberta-multitask",num_layers = 7, verbose=True)
    return {"P":float(P.mean()),"R":float(R.mean()),"F1":float(F1.mean())}

def get_bleu_score(path):
    df = pd.read_csv(path)
    print(len(df))
    references = []
    candidates = []
    for idx, row in df.iterrows():
        ref, cand = row["answer"],row["generated_answer"]
        ref = tokenizer.encode(text = ref,add_special_tokens= False)
        cand =  tokenizer.encode(text = cand,add_special_tokens= False)
        references.append([ref])
        candidates.append(cand)
    score4 = corpus_bleu(references, candidates,weights=(1,0,0,0))
    return score4

def get_rouge_score(path):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"])
    df = pd.read_csv(path)
    references, candidates = [],[]
    total = 0
    for idx, row in df.iterrows():
        ref, cand = row["answer"],row["generated_answer"]
        score = scorer.score(ref, cand)
        total += score["rouge1"].fmeasure
    return total/len(df)
  
if __name__ == "__main__":
    path = "/Users/daum0604/Desktop/output/BM25.csv"
    print(f"bleu-1:{get_bleu_score(path)} ")
    print(f"rouge-1:{get_rouge_score(path)} ")
    print(f"bert_score:{get_bert_score(path)} ")
    
    path = "/Users/daum0604/Desktop/output/post_retriever.csv"
    print(f"bleu-1:{get_bleu_score(path)} ")
    print(f"rouge-1:{get_rouge_score(path)} ")
    print(f"bert_score:{get_bert_score(path)} ")