from peft import LoraConfig
from transformers import BitsAndBytesConfig, TrainingArguments
import torch

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype= torch.float16,
    bnb_4bit_use_double_quant=True,
)
def set_trainer(config):
    training_arguments = TrainingArguments(
            output_dir=config["output_dir"],
            per_device_train_batch_size=config["train_batch_size"],
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            optim="paged_adamw_32bit",
            num_train_epochs=config["max_epoch"],
            evaluation_strategy="steps",
            eval_steps=0.2,
            logging_steps=1,
            warmup_steps=10,
            logging_strategy="steps",
            learning_rate=config["learning_rate"],
            fp16=False,
            bf16=False,
            group_by_length=True,
            report_to="wandb"
        )
    return training_arguments