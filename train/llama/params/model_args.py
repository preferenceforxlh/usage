from dataclasses import dataclass, field
from typing import Optional



@dataclass
class ModelArguments:
    model_name_or_path:Optional[str] = field(
        default=None, metadata={"help": "训练集的路径"}
    )
    flash_attn:Optional[bool] = field(
        default=True, metadata={"help": "Enable FlashAttention2"}
    )
    load_in_4bit:Optional[bool] = field(
        default=False, metadata={"help": "是否4bit加载模型"}
    )
    lora_rank:Optional[int] = field(
        default=0, metadata={"help": "Lora rank"}
    )
    lora_alpha:Optional[int] = field(
        default=64, metadata={"help": "Lora alpha"}
    )
    target_modules:Optional[str] = field(
        default="all-linear", metadata={"help": "lora目标模块"}
    )
    modules_to_save:Optional[str] = field(
        default=None, metadata={"help": "还需要保存的模块"}
    )
    lora_dropout:Optional[float] = field(
        default=0.0, metadata={"help": "lora dropout"}
    )