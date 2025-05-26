import json
import os
import torch
#import wandb
import random
import argparse
from datasets import load_dataset

from safetensors import safe_open

from transformers import (
    GenerationConfig,
    Llama4Config,
    Llama4ForConditionalGeneration,
    Llama4ImageProcessorFast,
    Llama4Processor,
    Llama4TextConfig,
    Llama4VisionConfig,
    PreTrainedTokenizerFast,
)
from transformers.models.llama.configuration_llama import LlamaConfig

from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM

from modules.model.llama_eagle import LlamaForCausalLMEagle
from modules.data.data import (
    EagleLocalDataset,
    DataCollatorWithPadding,
    AddUniformNoise,
    list_local_files,
)
from modules.trainer.trainer import EagleTrainer


parser = argparse.ArgumentParser(description="Train BaldEagle model")
parser.add_argument("--data_dir", type=str, default="outdir0", help="Directory containing the generated data")
parser.add_argument("--model_path", type=str, default="/root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/", help="Path to the base Llama model")
parser.add_argument("--generate_on_fly", action="store_true", help="Generate data on the fly instead of loading from disk")
parser.add_argument("--max_sharegpt_samples", type=int, default=100, help="Maximum number of ShareGPT samples to download")
parser.add_argument("--max_ultrachat_samples", type=int, default=100, help="Maximum number of UltraChat samples to download")
args = parser.parse_args()

wandb_run_name="test"
#wandb.init(project="BaldEagle")
#wandb_run_name = wandb.run.name

path = args.model_path

# -------------------------------- Load original Llama weights --------------------------------

with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
    index_json = json.loads(f.read())
    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
    lm_head_path = index_json["weight_map"]["lm_head.weight"]

with safe_open(os.path.join(path, emb_path), framework="pt", device="cpu") as f:
    tensor_slice = f.get_slice("model.embed_tokens.weight")
    vocab_size, hidden_dim = tensor_slice.get_shape()
    tensor = tensor_slice[:, :hidden_dim]

with safe_open(os.path.join(path, lm_head_path), framework="pt", device="cpu") as f:
    lm_head_weights = f.get_slice("lm_head.weight")[:, :]


# -------------------------------- Create draft model + tokenizer + head --------------------------------

tokenizer = AutoTokenizer.from_pretrained(path)
tokenizer.pad_token = tokenizer.eos_token

target_model = AutoModelForCausalLM.from_pretrained(
    path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

model_args = LlamaConfig(
    vocab_size=vocab_size,
    hidden_size=hidden_dim,
    intermediate_size=14336,
    num_hidden_layers=1,
    bos_token_id=128000,
    eos_token_id=[128001, 128008, 128009],
    num_key_value_heads=8,
    num_attention_heads=32,
    tie_word_embeddings=False,
)
#model_args = AutoConfig.from_pretrained("config.json", local_files_only=True)
#model_args = Llama4Config(
#    num_hidden_layers=1,
#)

draft_model = LlamaForCausalLMEagle(model_args)
draft_model.load_embedding_weights(tensor)
draft_model.embed_tokens.weight.requires_grad = False

# Load head
head = torch.nn.Linear(model_args.hidden_size, model_args.vocab_size, bias=False)
with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
    index_json = json.loads(f.read())
    head_path = index_json["weight_map"]["lm_head.weight"]
with safe_open(os.path.join(path, head_path), framework="pt", device="cpu") as f:
    tensor_slice = f.get_slice("lm_head.weight")
    vocab_size, hidden_dim = tensor_slice.get_shape()
    tensor = tensor_slice[:, :hidden_dim].float()

head.weight.data = tensor
head.eval()

max_len=100
eagle_train_dataset = EagleLocalDataset(
    target_model=target_model,
    transform=AddUniformNoise(std=0.5),
    tokenizer=tokenizer,
)
eagle_test_dataset = EagleLocalDataset(
    target_model=target_model,
    tokenizer=tokenizer,
    is_eval=True,
)

eagle_collator = DataCollatorWithPadding()

# -------------------------------- Train --------------------------------

training_args = TrainingArguments(
    output_dir=f"./hf_trainer_output_dir/{wandb_run_name}/",
    num_train_epochs=10,
    gradient_accumulation_steps=16,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    remove_unused_columns=False,
    bf16=True,
    fp16=False,
    dataloader_num_workers=0,
    warmup_ratio=0.01,
    learning_rate=1e-4,  # 1e-3
    lr_scheduler_type="constant",  # Placeholder, we override it in the trainer
    max_grad_norm=0.5,  # 1
    adam_beta1=0.9,  # 0.9
    adam_beta2=0.95,  # 0.999
    weight_decay=1e-2,
    eval_strategy="steps",
    logging_steps=32,
    eval_steps=64,
    save_strategy="steps",
    save_steps=0.1,  # saves every 10% of training
    save_total_limit=3,
)


trainer = EagleTrainer(
    model=draft_model,
    head=head,
    args=training_args,
    train_dataset=eagle_train_dataset,
    eval_dataset=eagle_test_dataset,
    data_collator=eagle_collator,
    min_lr_ratio=0.5,  # Custmer lr scheduler param
)

trainer.train()
