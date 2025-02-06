import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
from pathlib import Path
from datasets import load_dataset,concatenate_datasets
from llama.utils import PrintUtil


class PretrainDataset(Dataset):
    def __init__(
        self,
        dataset,
        max_examples,
        tokenizer,
        cutoff_length,
        input_key
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_examples = max_examples
        self.cutoff_length = cutoff_length
        self.input_key = input_key
    
        if os.path.isdir(dataset):
            path = Path(dataset)
            files = [os.path.join(path,file.name) for file in path.glob("*.json")]
            PrintUtil.print_rank_0(f"Training files: {' '.join(files)}")
            all_datasets = []
            for file in files:
                dataset = load_dataset("json", data_files=file)
                all_datasets.append(dataset)
            datasets = concatenate_datasets(all_datasets)
            self.dataset = datasets["train"]
            self.dataset = self.dataset.select(range(min(self.max_examples,len(self.dataset))))
        else:
            dataset = load_dataset("json", data_files=dataset)
            self.dataset = dataset["train"]
            self.dataset = self.dataset.select(range(min(self.max_examples,len(self.dataset))))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        text = self.dataset[index][self.input_key]
        text = text + self.tokenizer.eos_token
        inputs = self.tokenizer(
            text,
            max_length=self.cutoff_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        # 避免EOS token被截断
        inputs["input_ids"][0][-1] = self.tokenizer.eos_token_id
        inputs["attention_mask"][0][-1] = True
        inputs["input_ids"] = inputs["input_ids"].squeeze(0)
        inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)
        return inputs # [seq_len]

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
        # 进行pad
        input_ids = self.zero_pad_sequences(input_ids,"right",value=self.tokenizer.pad_token_id)
        attention_mask = self.zero_pad_sequences(attention_mask,"right",value=0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }