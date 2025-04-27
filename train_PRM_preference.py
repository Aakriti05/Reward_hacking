import os
os.environ["DEEPSPEED_USE_MPI"] = "0"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"

from datasets import load_dataset
from trl import PRMConfig, PRMTrainer
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModel
import torch
cache_dir = "/cmlscratch/agrawal5/cache"

model_name = "Qwen/Qwen2.5-Math-PRM-7B"
device = "auto"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
print("tokenizer loaded")

model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=cache_dir,
    low_cpu_mem_usage=True
)

model.gradient_checkpointing_enable()

print("model loaded")
train_dataset = load_dataset("trl-lib/math_shepherd", split="train[:1]")
print("data loaded")

# Define config
training_args = PRMConfig(
    output_dir="Qwen2.5-Math-PRM-7B-Math-Sheperd",
    logging_steps=10,
    per_device_train_batch_size=1,       # Smaller batch size
    gradient_accumulation_steps=1,       # Accumulate gradients to simulate larger batch
    fp16=True, 
    deepspeed="deepspeed_config_zero1.json",
    save_total_limit=1,
)

print("config set")
trainer = PRMTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
print("starting training")
trainer.train()