import os
import random
import shutil
from abc import ABC
from collections import defaultdict
from datetime import timedelta

import deepspeed
import numpy as np
import torch
import torch.nn as nn
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from peft import PeftModel, get_peft_model_state_dict
from torch import distributed as dist
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from llama.utils import PrintUtil

from .deepspeed_utils import (
    _z3_params_to_fetch,
    get_optimizer_grouped_parameters,
    get_train_ds_config,
)

class DeepspeedStrategy(ABC):
    """
    The strategy for training with Accelerator.
    """

    def __init__(self,args=None) -> None:
        super().__init__()
        self.args = args
        self.stage = args.zero_stage
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.bf16 = args.bf16
        self.seed = args.seed
        self.max_norm = args.max_grad_norm
        self.adam_offload = getattr(args, "adam_offload", False)
        self.param_offload = getattr(args, "param_offload", False)
        self.zpg = getattr(args, "zpg", 1)
        self.grad_accum_dtype = getattr(args, "grad_accum_dtype", None)
        # overlap_comm
        self.overlap_comm = getattr(args, "overlap_comm", False)

        self.time_steps = defaultdict(int)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_distributed(self, timeout=timedelta(minutes=60)) -> None:
        self.set_seed(self.seed)

        if self.args.local_rank == -1: 
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            self.device = torch.device("cuda",self.args.local_rank)
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            deepspeed.init_distributed(timeout=timeout)
        
        self.world_size = dist.get_world_size()
        # 计算每个batch处理的样本数
        self.train_batch_size = self.per_device_train_batch_size * self.gradient_accumulation_steps * self.world_size
    
    def to_device(self,batch):
        output = {}
        for k, v in batch.items():
            try:
                output[k] = v.to(self.device)
            except:
                output[k] = v
        return output

    def create_optimizer(self, model, **kwargs) -> Optimizer:
        # Optimizer
        PrintUtil.print_rank_0(f"Creating optimizer with kwargs: {kwargs}")
        AdamOptimizer = DeepSpeedCPUAdam if self.adam_offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(model, kwargs["weight_decay"])
        optim = AdamOptimizer(optim_params, **kwargs)
        return optim

    def setup_dataloader(
        self,
        dataset,
        batch_size: int,
        collate_fn=None,
        is_train=True
    ):
        if self.args.local_rank == -1:
            sampler = RandomSampler(dataset) if is_train else SequentialSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)
            
        return DataLoader(
            dataset,
            shuffle=(sampler is None),
            collate_fn=collate_fn,
            sampler=sampler,
            batch_size=batch_size,
            pin_memory=True
        )

    def _unwrap_model(self, model) -> nn.Module:
        if hasattr(model, "module"):
            return model.module
        else:
            return model

    def prepare(self,model,optimizer,scheduler):
        ds_config = self.get_ds_train_config()

        model, optim, _, scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=ds_config,
            args={"local_rank": self.args.local_rank},
            dist_init_required=True,
        )
        return model, optim, scheduler

    def get_ds_train_config(self):
        # DS Config
        ds_config = get_train_ds_config(
            offload=self.param_offload,
            adam_offload=self.adam_offload,
            stage=self.stage,
            bf16=self.bf16,
            max_norm=self.max_norm,
            zpg=self.zpg,
            grad_accum_dtype=self.grad_accum_dtype,
            overlap_comm=self.overlap_comm,
        )
        PrintUtil.print_rank_0(f"DeepSpeed Config\n: {ds_config}")
        ds_config["train_micro_batch_size_per_gpu"] = self.per_device_train_batch_size
        ds_config["train_batch_size"] = self.train_batch_size 

        return ds_config

    def save_model(self, model: nn.Module, tokenizer, output_dir, **kwargs) -> None:
        if PrintUtil.is_rank_0():
            os.makedirs(output_dir, exist_ok=True)

        # save model weights for ZeRO2/3
        model_to_save = self._unwrap_model(model)

        # gather parameters
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            # only gather z3 params
            params_to_fetch = _z3_params_to_fetch([v])
            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                vv = v.data.cpu()
                if PrintUtil.is_rank_0():
                    output_state_dict[k] = vv

        if PrintUtil.is_rank_0():
            state_dict = model_to_save.state_dict()

            # copy named_buffers with `persistent=True`
            for k, v in model_to_save.named_buffers():
                if k not in state_dict:
                    continue
                vv = v.data.cpu()
                output_state_dict[k] = vv

            state_dict_keys = set(state_dict.keys())
            output_state_dict_keys = set(output_state_dict.keys())

            # corner case for tie_word_embeddings, such as Qwen2-0.5B
            if getattr(model_to_save.config, "tie_word_embeddings", False) and "lm_head.weight" in state_dict_keys:
                state_dict_keys.remove("lm_head.weight")

            assert state_dict_keys.issubset(
                output_state_dict_keys
            ), f"mismatch keys {output_state_dict_keys.symmetric_difference(state_dict_keys)}"

            # only save peft weights https://github.com/microsoft/DeepSpeed/issues/4295
            if isinstance(model_to_save, PeftModel):
                model_to_save.save_pretrained(output_dir, **kwargs)
                if self.stage == 3:
                    torch.save(
                        get_peft_model_state_dict(model_to_save, output_state_dict),
                        os.path.join(output_dir, "adapter_model.bin"),
                    )
                    filename = os.path.join(output_dir, "adapter_model.safetensors")
                    if os.path.exists(filename):
                        os.remove(filename)
            else:
                # save model
                model_to_save.save_pretrained(output_dir, state_dict=output_state_dict, **kwargs)

            # save config
            output_config_file = os.path.join(output_dir, "config.json")
            model_to_save.config.to_json_file(output_config_file)
            # save tokenizer
            tokenizer.save_pretrained(output_dir)

            # for models not in AutoModel, copy python module files
            train_from_model_path = model_to_save.config._name_or_path
            if os.path.exists(train_from_model_path):
                for filename in os.listdir(train_from_model_path):
                    if filename.endswith(".py"):
                        shutil.copy(os.path.join(train_from_model_path, filename), os.path.join(output_dir, filename))

    def all_reduce(self, data, op="mean"):
        assert op in ("mean", "max", "sum")
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_reduce(v, op)
            return ret
        else:
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"

            if is_cpu_tensor:
                data = data.to(self.device)
            if op == "mean":
                data /= self.world_size
            dist.all_reduce(data, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
            if is_cpu_tensor:
                data = data.cpu()
            return data.item() if not is_tensor else data

    def all_gather(self, data):
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_gather(v)
            return ret
        else:
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
            is_cpu_tensor = data.device.type == "cpu"

            ret = [torch.zeros_like(data).to(self.device) for _ in range(self.world_size)]
            dist.all_gather(ret, data.to(self.device))
            return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)
