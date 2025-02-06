import torch
from torch import distributed as dist   
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import get_peft_model
from peft import LoraConfig, TaskType,prepare_model_for_kbit_training


class PrintUtil:
    @classmethod
    def is_rank_0(cls):
        return not dist.is_initialized() or dist.get_rank() == 0
    
    @classmethod
    def print_rank_0(cls,*msg):
        if PrintUtil.is_rank_0():
            print(*msg)


def get_model_and_tokenizer(args,padding_side="right"):
    # 加载模型
    if args.load_in_4bit:
        assert args.bf16 ,"we only support bnb_4bit_compute_dtype = bf16"
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else "auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.flash_attn else "sdpa"
    )

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model=model,use_gradient_checkpointing=args.gradient_checkpointing)

    # lora
    if args.lora_rank > 0:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            bias="none",
        )
        model = get_peft_model(model,lora_config)
        if PrintUtil.is_rank_0():
            model.print_trainable_parameters()
    
    model.config.use_cache = False
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = padding_side # 修改tokenizer的pad方向
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        PrintUtil.print_rank_0("pad token id",model.config.pad_token_id)

    return model,tokenizer


def get_strategy(args):
    from llama.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(args=args)
    return strategy