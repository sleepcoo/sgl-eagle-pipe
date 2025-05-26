import torch
import os
import json
import re
import multiprocessing
from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set multiprocessing start method to 'spawn' for CUDA compatibility

multiprocessing.set_start_method('spawn', force=True)

import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

def list_local_files(path, suffixes=[".ckpt"]):
    datapaths = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapaths.append(file_path)

    # Filter out files that don't end with the suffixes (ie. when there's a HuggingFace .cache folder)
    for suffix in suffixes:
        datapaths = [f_name for f_name in datapaths if f_name.endswith(suffix)]

    return datapaths


def list_hf_files(repo, suffixes=[".ckpt"]):
    hf_fs = HfFileSystem()
    datapaths = []
    print(
        f"Listing files in {repo}. This is expected to take ~2 min for ShareGPT (70k files)."
    )
    for path, _, files in tqdm(hf_fs.walk(repo)):
        for file in files:
            datapaths.append(path + "/" + file)

    # Filter out files that don't end with the suffixes (ie. when there's a HuggingFace .cache folder)
    for suffix in suffixes:
        datapaths = [f_name for f_name in datapaths if f_name.endswith(suffix)]

    print(f"Found {len(datapaths)} files")
    return datapaths


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        # Follow EAGLE uniform noise
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data
system_message = {
    "role": "system",
    "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
}

def format_conversation_sharegpt(row, dataset_column="conversations"):
    messages = [system_message]
    current_role = None
    for message in row[dataset_column]:
        if message["from"] == "human":
            messages.append({
                "role": "user",
                "content": message["value"]}
            )
        elif message["from"] == "gpt":
            messages.append({
                "role": "assistant",
                "content": message["value"]}
            )
        else:
            raise ValueError(f"Unknown role: {message['from']}")

        if current_role is None:
            current_role = messages[-1]["role"]
        else:
            assert current_role != messages[-1]["role"], f"Conversation has incorrect role order"
            current_role = messages[-1]["role"]

    return {"messages": messages}


assistant_header = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
user_header = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
def tokenize_conversation(row, tokenizer, col="messages"):
    formatted_conversation = tokenizer.apply_chat_template(
        row[col], tokenize=False, add_generation_prompt=False
    )

    encoding = tokenizer(formatted_conversation, return_offsets_mapping=True)
    input_ids = encoding.input_ids
    offsets = encoding.offset_mapping
    loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

    # Find spans of assistant responses using regex
    assistant_pattern = (
        re.escape(assistant_header) + r"(.*?)(?=" + re.escape(user_header) + "|$)"
    )
    for match in re.finditer(assistant_pattern, formatted_conversation, re.DOTALL):
        # Assistant response text span (excluding assistant_header itself)
        assistant_start_char = match.start(1)
        assistant_end_char = match.end(1)

        # Mark tokens overlapping with assistant response
        for idx, (token_start, token_end) in enumerate(offsets):
            # Token is part of the assistant response span
            if token_end <= assistant_start_char:
                continue  # token before assistant text
            if token_start > assistant_end_char:
                continue  # token after assistant text
            loss_mask[idx] = 1

    return {
        "conversation_str": formatted_conversation,
        "input_ids": input_ids,
        "loss_mask": loss_mask,
    }


class EagleLocalDataset(IterableDataset):
    def __init__(
        self,
        target_model: str,
        tokenizer,
        transform=None,
        max_len: int = 2048,
        seed: int = 42,
        device: str = "cuda",
        is_eval:bool=False,
    ):
        super().__init__()
        self.transform = transform
        self.max_len = max_len
        self.seed = seed
        self.device = device
        
        # Initialize tokenizer and model
        self.tokenizer =tokenizer
        self.target_model = target_model
        self.target_model.eval()
        
        # Load and process dataset
        self.dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
        self.dataset = self.dataset.select(range(0, 10000))
        self.dataset = self.dataset.shuffle(seed=seed)
        self.dataset = self.dataset.map(format_conversation_sharegpt)
        self.dataset = self.dataset.map(
            tokenize_conversation,
            fn_kwargs={"tokenizer": self.tokenizer}
        )
        self.dataset = self.dataset.remove_columns(
            [
                col
                for col in self.dataset.column_names
                if col not in ["input_ids", "loss_mask", "conversation_str"]
            ]
        )
        self.dataset.set_format(type="torch")
        
        # Split dataset into train and validation
        train_size = int(len(self.dataset) * 0.95)
        if is_eval:
            self.dataset = self.dataset.select(range(train_size,  int(len(self.dataset))))  # Take first 100 examples from validation set
        else:
            self.dataset = self.dataset.select(range(train_size))  # Take first 95% for training
        
    def __iter__(self):
        for row in self.dataset:
            with torch.no_grad():
                outputs = self.target_model(
                    row["input_ids"].unsqueeze(0),
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1]

            data = {
                "input_ids": row["input_ids"],
                "loss_mask": row["loss_mask"],
                "hidden_state": hidden_states,
            }
            new_data = {}

            # Squeeze due to our data generation script adding a batch dimension
            hidden_state = data["hidden_state"].squeeze(0)[: self.max_len][None, :]

            input_ids = data["input_ids"][: self.max_len][None, :]
            loss_mask = data["loss_mask"][: self.max_len][None, :]

            length = hidden_state.shape[1]
            attention_mask = [1] * length
            loss_mask = loss_mask[0].tolist()
            loss_mask[-1] = 0

            input_ids_target = input_ids[:, 1:]
            zeropadding = torch.tensor([[0]])
            input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

            target = hidden_state[:, 1:, :]
            zeropadding = torch.zeros(1, 1, target.shape[2])
            target = torch.cat((target, zeropadding), dim=1)
            loss_mask[-1] = 0
            new_data["attention_mask"] = attention_mask
            new_data["loss_mask"] = loss_mask
            new_data["target"] = target
            new_data["hidden_state_big"] = hidden_state
            new_data["input_ids"] = input_ids_target


            if self.transform:
                new_data = self.transform(new_data)
            yield new_data
            
    def __len__(self):
        return len(self.dataset)


class DataCollatorWithPadding:
    # Copied from https://github.com/SafeAILab/EAGLE/blob/main/eagle/train/main.py#L178

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features):
        max_length = max(item["hidden_state_big"].shape[1] for item in features)
        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_hidden_states = torch.cat(
            [
                self.paddingtensor(item["hidden_state_big"], max_length)
                for item in features
            ]
        )
        batch_target = torch.cat(
            [self.paddingtensor(item["target"], max_length) for item in features]
        )
        batch_loss_mask = torch.tensor(
            [
                item["loss_mask"] + [0] * (max_length - len(item["loss_mask"]))
                for item in features
            ]
        )
        batch_attention_mask = torch.tensor(
            [
                item["attention_mask"]
                + [0] * (max_length - len(item["attention_mask"]))
                for item in features
            ]
        )
        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch
