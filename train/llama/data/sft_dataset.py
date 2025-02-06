import os
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset
from datasets import load_dataset,concatenate_datasets
from llama.utils import PrintUtil

class SFTDataset(Dataset):
    def __init__(
        self,
        dataset,
        max_examples,
        tokenizer,
        cutoff_length,
        prompt_key,
        query_key,
        output_key
    ):
        super().__init__()
        self.dataset = dataset
        self.max_examples = max_examples
        self.tokenizer = tokenizer
        self.cutoff_length = cutoff_length
        self.prompt_key = prompt_key
        self.query_key = query_key
        self.output_key = output_key
    
        if os.path.isdir(dataset):
            path = Path(dataset)
            files = [os.path.join(path,file.name) for file in path.glob("*.json")]
            PrintUtil.print(f"Training files: {' '.join(files)}")
            all_datasets = []
            for file in files:
                dataset = load_dataset("json", data_files=file)
                all_datasets.append(dataset)
            datasets = concatenate_datasets(all_datasets)
            self.dataset = datasets["train"]
        else:
            dataset = load_dataset("json", data_files=dataset)
            self.dataset = dataset["train"]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        prompt = data[self.prompt_key]
        if data[self.query_key] != "":
            prompt = prompt + "\n" + data[self.query_key]
        
        prompt_message = [{"role":"user","content":prompt}]
        response_message = [{"role":"assistant","content":data[self.output_key]}]

        prompt = self.tokenizer.apply_chat_template(prompt_message,tokenize=False,add_generation_prompt=True)
        response = self.tokenizer.apply_chat_template(prompt_message + response_message,tokenize=False)[len(prompt):]

        prompt_tokenized = self.tokenizer(
            prompt,
            max_length=self.cutoff_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        response_tokenized = self.tokenizer(
            response,
            max_length=self.cutoff_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        all_input_ids = torch.cat([prompt_tokenized["input_ids"],response_tokenized["input_ids"]],dim=-1)
        all_attention_mask = torch.cat([prompt_tokenized["attention_mask"],response_tokenized["attention_mask"]],dim=-1)
        if all_input_ids.shape[-1] > self.cutoff_length:
            all_input_ids = all_input_ids[:,:self.cutoff_length]
            all_attention_mask = all_attention_mask[:,:self.cutoff_length]
        
        # 避免eos token被截断
        all_input_ids[0][-1] = self.tokenizer.eos_token_id
        all_attention_mask[0][-1] = True

        return {
            "input_ids": all_input_ids.squeeze(0),
            "attention_mask": all_attention_mask.squeeze(0),
            "prompt_length": prompt_tokenized["attention_mask"].int().sum().item()
        }

    def zero_pad_sequences(self,sequences, side: str = "left", value=0):
        assert side in ("left", "right")
        max_len = max(seq.size(-1) for seq in sequences)
        padded_sequences = []
        for seq in sequences:
            pad_len = max_len - seq.size(-1)
            padding = (pad_len, 0) if side == "left" else (0, pad_len)
            padded_sequences.append(F.pad(seq, padding, value=value))
        return torch.stack(padded_sequences, dim=0)

    def collate_fn(self,examples):
        input_ids = [example["input_ids"] for example in  examples]
        attention_mask = [example["attention_mask"] for example in examples]
        prompt_lengths = [example["prompt_length"] for example in examples]
        # 进行pad
        input_ids = self.zero_pad_sequences(input_ids,"right",value=self.tokenizer.pad_token_id)
        attention_mask = self.zero_pad_sequences(attention_mask,"right",value=0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_lengths": prompt_lengths
        }