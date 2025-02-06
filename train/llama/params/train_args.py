from dataclasses import dataclass, field
from transformers import TrainingArguments as HFTrainingArguments
from typing import Optional


@dataclass
class TrainingArguments(HFTrainingArguments):
    zero_stage:Optional[int] = field(
        default=2, metadata={"help": "DeepSpeed ZeRO stage"}
    )
    zpg:Optional[int] = field(
        default=1, metadata={"help": "ZeRO++ max partition size"}
    )
    adam_offload:Optional[bool] = field(
        default=False, metadata={"help": "Offload Adam Optimizer"}
    )
    param_offload:Optional[bool] = field(
        default=False, metadata={"help": "Offload model param"}
    )
    grad_accum_dtype:Optional[str] = field(
        default=None, metadata={"help": "Adam grad accum data type"}
    )
    overlap_comm:Optional[bool] = field(
        default=True
    )
    # swanlab config
    swanlab_api_key:Optional[str] = field(
        default=None
    )
    swanlab_project_name:Optional[str] = field(
        default="llama"
    )
    swanlab_experiment_name:Optional[str] = field(
        default="pretrain"
    )