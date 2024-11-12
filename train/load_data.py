import pandas as pd
from datasets import Dataset

TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{context}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{answer}<|eot_id|>"""

class MyDatasets:
  def __init__(self,max_seq = None):
    self.PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 법률 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''
    self.data = []
    self.max_seq = max_seq
    
  def format_chat_template(self,data_frame):
    for idx,row in data_frame.iterrows():
        if not self.max_seq or int(row["token_counts"]) <= self.max_seq:
          self.data.append({"text":TEMPLATE.format(context = self.PROMPT,question = row["question"],answer = row["answer"])})
        
  def __load_csv(self,path):
    return pd.read_csv(path)
  def build_dataset(self,path):
    data_frame = self.__load_csv(path)
    self.format_chat_template(data_frame)
    data = Dataset.from_list(self.data)
    return  data
  
  def build_inference(self,path):
    data_frame = self.__load_csv(path)
    for idx,row in data_frame.iterrows():
        self.data.append({"question":row["question"],"answer":row["answer"]})
    data = Dataset.from_list(self.data)
    return  data