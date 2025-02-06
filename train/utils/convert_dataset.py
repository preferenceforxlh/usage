import json
from pathlib import Path
import os
from datasets import load_dataset
from collections import defaultdict

def convert_llamafactory_data_json(dataset_dir,output_dir):
    path = Path(dataset_dir)
    output_path = Path(output_dir)
    files = [file.name for file in path.glob("*.json")]
    for file in files:
        data_file = os.path.join(path, file)
        raw_dataset = load_dataset("json", data_files=data_file,keep_in_memory=False)
        dataset = raw_dataset['train']
        output = []
        for i in range(len(dataset)):
            output.append(dataset[i])
        output_file = os.path.join(output_path, file.replace(".json", "_llamafactory.json"))
        with open(output_file, 'w',encoding='utf-8') as f:
            f.write(json.dumps(output, ensure_ascii=False, indent=4))

def convert_llamafactory_data_json_rw(dataset_dir,output_dir):
    path = Path(dataset_dir)
    output_path = Path(output_dir)
    files = [file.name for file in path.glob("*.json")]
    for file in files:
        data_file = os.path.join(path, file)
        raw_dataset = load_dataset("json", data_files=data_file,keep_in_memory=False)
        dataset = raw_dataset['train']
        output = []
        for i in range(len(dataset)):
            output.append({"instruction":dataset[i]["question"],"input":"","chosen":dataset[i]["response_chosen"],
                           "rejected":dataset[i]["response_rejected"]})
        output_file = os.path.join(output_path, file.replace(".json", "_llamafactory.json"))
        with open(output_file, 'w',encoding='utf-8') as f:
            f.write(json.dumps(output, ensure_ascii=False, indent=4))


def convert_openrlhf_sft_data_json(dataset_dir,output_dir):
    path = Path(dataset_dir)
    output_path = Path(output_dir)
    files = [file.name for file in path.glob("*.json")]
    for file in files:
        data_file = os.path.join(path, file)
        raw_dataset = load_dataset("json", data_files=data_file,keep_in_memory=False)
        dataset = raw_dataset['train']
        output = []
        for i in range(len(dataset)):
            data = dataset[i]
            new_data = defaultdict(list)
            new_data["prompt"].append(
                {"role":"user","content":data["instruction"] + "\n" + data["input"] if data["input"] is not None else data["instruction"]} 
            )
            new_data["prompt"].append(
                {"role":"assistant","content":data["output"]}
            )
            output.append(new_data)
        output_file = os.path.join(output_path, file.replace(".json", "openrlhf.json"))
        with open(output_file, 'w',encoding='utf-8') as f:
            f.write(json.dumps(output, ensure_ascii=False, indent=4))


def convert_openrlhf_rw_data_json(dataset_dir,output_dir):
    path = Path(dataset_dir)
    output_path = Path(output_dir)
    files = [file.name for file in path.glob("*.json")]
    for file in files:
        data_file = os.path.join(path, file)
        raw_dataset = load_dataset("json", data_files=data_file,keep_in_memory=False)
        dataset = raw_dataset['train']
        output = []
        for i in range(len(dataset)):
            data = dataset[i]
            new_data = defaultdict(list)
            new_data["chosen"].append(
                {"role":"user","content":data["question"]} 
            )
            new_data["chosen"].append(
                {"role":"assistant","content":data["response_chosen"]}
            )
            new_data["rejected"].append(
                {"role":"user","content":data["question"]} 
            )
            new_data["rejected"].append(
                {"role":"assistant","content":data["response_rejected"]}
            )
            output.append(new_data)
        output_file = os.path.join(output_path, file.replace(".json", "openrlhf.json"))
        with open(output_file, 'w',encoding='utf-8') as f:
            f.write(json.dumps(output, ensure_ascii=False, indent=4))

