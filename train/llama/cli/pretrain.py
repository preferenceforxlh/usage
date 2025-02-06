import os
import math
import argparse

from transformers.trainer import get_scheduler

from llama.utils import get_strategy,get_model_and_tokenizer,PrintUtil
from llama.data import PretrainDataset
from llama.trainer import PretrainTrainer


def train(args):
    # config strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # 加载模型和tokenizer
    model,tokenizer = get_model_and_tokenizer(args)
    # 打印模型
    PrintUtil.print_rank_0("model:\n",model)

    # 是否开启梯度检查点
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # 创建optimizer
    optimizer = strategy.create_optimizer(model,lr=args.learning_rate,
                                      weight_decay=args.weight_decay,
                                      betas=args.adam_betas)
   # 创建数据集
    PrintUtil.print_rank_0("*" * 10 + " loading train datasets " + "*" * 10)
    train_dataset = PretrainDataset(args.train_dataset,max_examples=args.max_examples,tokenizer=tokenizer,
                                    cutoff_length=args.max_len,input_key=args.input_key)
    PrintUtil.print_rank_0(f"Num train_samples  {len(train_dataset)}")
    PrintUtil.print_rank_0("Training example:")
    PrintUtil.print_rank_0(tokenizer.decode(train_dataset[0]['input_ids']))

    PrintUtil.print_rank_0("*" * 10 + " loading eval datasets " + "*" * 10)
    eval_dataset = PretrainDataset(args.eval_dataset,max_examples=args.max_examples,tokenizer=tokenizer,
                                    cutoff_length=args.max_len,input_key=args.input_key)
    PrintUtil.print_rank_0(f"Num eval_samples  {len(eval_dataset)}")
    PrintUtil.print_rank_0("Eval example:")
    PrintUtil.print_rank_0(tokenizer.decode(eval_dataset[0]['input_ids']))
    
    # 创建dataloader
    ## 如果是分布式训练，不需要传入sampler，如果是单机训练，需要传入sampler
    train_dataloader = strategy.setup_dataloader(
        dataset=train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=train_dataset.collate_fn,
        is_train=True,
    )
    eval_dataloader = strategy.setup_dataloader(
        dataset=eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=eval_dataset.collate_fn,
        is_train=False,
    )

    # 创建scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / strategy.train_batch_size)
    max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=math.ceil(max_steps * args.warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    # deepspeed包装
    model, optimizer, lr_scheduler = strategy.prepare(model=model,optimizer=optimizer,scheduler=lr_scheduler)

    # 开始训练
    PrintUtil.print_rank_0("***** Running training *****", args.local_rank)
    PrintUtil.print_rank_0("  Num examples = ", len(train_dataset))
    PrintUtil.print_rank_0("  Num Epochs = ", args.num_train_epochs)
    PrintUtil.print_rank_0("  Instantaneous batch size per device = ", args.per_device_train_batch_size)
    PrintUtil.print_rank_0("  Total train batch size (w. parallel, distributed & accumulation) = ", strategy.train_batch_size)
    PrintUtil.print_rank_0("  Gradient Accumulation steps = ", args.gradient_accumulation_steps)
    PrintUtil.print_rank_0("  Total optimization steps = ", max_steps)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化trainer
    trainer = PretrainTrainer(
        args=args,
        strategy=strategy,
        model=model,
        optim=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=lr_scheduler,
        max_norm=args.max_grad_norm,
        batch_size=strategy.train_batch_size,
        max_epochs=args.num_train_epochs,
        tokenizer=tokenizer
    )

    trainer.fit(args=args,num_update_steps_per_epoch=num_update_steps_per_epoch)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.output_dir)
    PrintUtil.print_rank_0("***** Finish training *****")

if __name__ == "__main__":
    # 创建parser
    parser = argparse.ArgumentParser()
    
    # 配置数据集相关的参数
    parser.add_argument("--train_dataset",type=str,default=None,help="训练集的路径")
    parser.add_argument("--eval_dataset",type=str,default=None,help="测试集的路径")
    parser.add_argument("--input_key",type=str,default="text",help="JSON dataset key")
    parser.add_argument("--max_len",type=int,default=1024,help="Max tokens for the samples")
    parser.add_argument("--max_examples",type=int,default=1000000,help="Max examples for the dataset")

    # 配置模型相关参数
    parser.add_argument("--model_name_or_path",type=str,default=None,help="模型的路径")
    parser.add_argument("--flash_attn",type=bool,action="store_true",help="Enable FlashAttention2")
    ## 低精度
    parser.add_argument("--load_in_4bit",type=bool,action="store_true",help="是否4bit加载模型")
    ## lora
    parser.add_argument("--lora_rank",type=int,default=0,help="Lora rank")
    parser.add_argument("--lora_alpha",type=int,default=64,help="Lora alpha")
    parser.add_argument("--target_modules",type=str,default="all-linear",help="lora目标模块")
    parser.add_argument("--modules_to_save",type=str,default=None,help="需要保存的embedding模块")
    parser.add_argument("--lora_dropout",type=float,default=0.0,help="lora dropout")

    # deepspeed训练相关参数
    parser.add_argument("--output_dir",type=str,default="./output",help="模型输出路径")
    parser.add_argument("--zero_stage",type=int,default=2,help="DeepSpeed ZeRO stage")
    parser.add_argument("--zpg",type=int,default=1,help="ZeRO++ max partition size")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--adam_offload",type=bool,action="store_true",help="Offload Adam Optimizer")
    parser.add_argument("--param_offload",type=bool,action="store_true",help="Offload model param")
    parser.add_argument("--grad_accum_dtype",type=str,default=None,help="Adam grad accum data type")
    parser.add_argument("--bf16", type=bool, action="store_true", help="Enable bfloat16")
    parser.add_argument("--overlap_comm",type=bool,action="store_true")
    parser.add_argument("--gradient_checkpointing",type=bool,action="store_true",help="gradient checkpointing")
    parser.add_argument("--gradient_checkpointing_use_reentrant", type=bool, action="store_true")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    parser.add_argument("--logging_steps",type=int,default=5,help="logging steps")
    parser.add_argument("--save_steps",type=int,default=500,help="save steps")
    parser.add_argument("--eval_steps",type=int,default=500,help="eval steps")
    parser.add_argument("--per_device_train_batch_size",type=int,default=2,help="per device train batch size")
    parser.add_argument("--per_device_eval_batch_size",type=int,default=2,help="per device eval batch size")
    parser.add_argument("--gradient_accumulation_steps",type=int,default=8,help="gradient accumulation steps")
    parser.add_argument("--max_grad_norm",type=float,default=1.0,help="max grad norm")
    parser.add_argument("--learning_rate",type=float,default=5e-5,help="learning rate")
    parser.add_argument("--lr_scheduler_type",type=str,default="cosine_with_min_lr",help="lr scheduler")
    parser.add_argument("--weight_decay",type=float,default=0.01,help="weight decay")
    parser.add_argument("--warmup_ratio",type=float,default=0.03,help="warmup ratio")
    parser.add_argument("--num_train_epochs",type=int,default=3,help="num train epochs")
    parser.add_argument("--seed",type=int,default=42,help="seed")
    
    # swanlab相关配置
    parser.add_argument("--swanlab_api_key",type=str,default="6IWs4PClMRoHWP0bU7Es2")
    parser.add_argument("--swanlab_project_name",type=str,default="llama")
    parser.add_argument("--swanlab_experiment_name",type=str,default="pretrain")

    args = parser.parse_args()

    # post-precessing
    if args.target_modules != "all-linear":
        args.target_modules =  args.target_modules.split(",")

    train(args)
