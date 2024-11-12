from argparse import ArgumentParser
from peft import LoraConfig
from transformers import (
    BitsAndBytesConfig, 
    TrainingArguments,   
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from trl import SFTTrainer, setup_chat_format
import torch
from train_env.arguments import BNB_CONFIG, set_trainer
from train_env.settings import WANDB, HUGGING_FACE

import load_data
import os,  wandb
from huggingface_hub import login
def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str, default="../data/train.csv")
    parser.add_argument('--model_name', type=str, default='MLP-KTLim/llama-3-Korean-Bllossom-8B')
    parser.add_argument('--output_dir', type=str, default="llama-3-8b-chat-lawyer-demo")
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--max_epoch', type=int, default=1)
    parser.add_argument('--max_seq', type=int, default=800)
    parser.add_argument('--seed', type=int, default=42, help='random seed for train valid split')
    args = parser.parse_args()
    return args

def train_init(project_name):
    #키 세팅
    wb_token = WANDB["key"]
    hf_token = HUGGING_FACE["key"]
    login(token = hf_token)
    wandb.login(key=wb_token)
    run = wandb.init(
        project= project_name, 
        job_type="training", 
        anonymous="allow"
    )
    
def train(config):
    #완디비 및 허깅페이스 설정
    train_init(config["output_dir"])
 
    
    #데이터 불러오기
    myData = load_data.MyDatasets(max_seq=config["max_seq"])
    dataset = myData.build_dataset(config["data_dir"])
    dataset = dataset.shuffle(seed=config["seed"]).train_test_split(test_size = 0.1)
    print(args)
   
    # 모델 불러오기
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config= BNB_CONFIG,
        device_map="auto",
        attn_implementation= "eager"
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )
    model = get_peft_model(model, peft_config)
    
    #토크나이저 불러오기
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    
    #학습 설정
    training_arguments = set_trainer(config=config)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        max_seq_length= config["max_seq"],
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing= False,
    )
    trainer.train()
    
    trainer.model.save_pretrained(config["output_dir"])
    trainer.model.push_to_hub(config["output_dir"], use_temp_dir=False)
    
args = parse_args()
train(args.__dict__)