import os
from abc import ABC

import torch
from torch.optim import Optimizer
from tqdm import tqdm

from llama.metric import GPTLMLoss
from llama.utils.deepspeed import DeepspeedStrategy
from llama.utils import PrintUtil


class SFTTrainer(ABC):
    def __init__(
        self,
        args,
        strategy:DeepspeedStrategy,
        model,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm: float = 1,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
    ) -> None:
        super().__init__()
        self.args = args
        self.strategy = strategy
        self.model = model
        self.optimizer = optim
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.max_norm = max_norm
        self.batch_size = batch_size
        self.epochs = int(max_epochs)
        self.tokenizer = tokenizer

        self.loss_fn = GPTLMLoss()

        # swanlab
        self._swanlab = None
        if PrintUtil.is_rank_0():
            import swanlab
            self._swanlab = swanlab
            swanlab.login(api_key=self.args.swanlab_api_key)
            swanlab.init(
                project=self.args.swanlab_project_name,
                experiment_name=self.args.swanlab_experiment_name,
            )

    def fit(self, args, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        step = 1 # global step

        epoch_bar = tqdm(
            range(0, self.epochs),
            desc="Train epoch",
            disable=not PrintUtil.is_rank_0(),
        )
        loss_sum = 0
        for epoch in range(0, self.epochs):
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not PrintUtil.is_rank_0(),
            )
            # train
            self.model.train()
            for i,batch_data in enumerate(self.train_dataloader):
                batch_data = self.strategy.to_device(batch_data)
                input_ids = batch_data["input_ids"]
                attention_mask = batch_data["attention_mask"]
                prompt_lengths = batch_data["prompt_lengths"]
                # forward
                output = self.model(input_ids,attention_mask=attention_mask)
                output.logits = output.logits.to(torch.float32)
                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    input_ids,
                    self.loss_fn.IGNORE_INDEX,
                )
                for label, source_len in zip(labels, prompt_lengths):
                    label[:source_len] = self.loss_fn.IGNORE_INDEX

                loss = self.loss_fn(output.logits, labels)
                self.model.backward(loss)
                self.model.step()

                loss_sum += loss.item()
                logs_dict = {
                    "loss": loss.item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }
                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.gradient_accumulation_steps == 0:
                    logs_dict["loss_mean"] = loss_sum / self.strategy.gradient_accumulation_steps
                    loss_sum = 0
                    global_step = step // self.strategy.gradient_accumulation_steps
                    self.save_logs_and_checkpoints(args, global_step,logs_dict)

                step += 1

            epoch_bar.update()

        if self._swanlab is not None and PrintUtil.is_rank_0():
            self._swanlab.finish()

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step,logs_dict={}):
        if global_step % args.logging_steps == 0:
            # swanlab
            if self._swanlab is not None and PrintUtil.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._swanlab.log(logs)

        # eval
        if global_step % args.eval_steps == 0:
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)

        # save ckpt
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            save_path = os.path.join(args.output_dir, f"{tag}_hf")
            self.strategy.save_model(self.model, self.tokenizer, save_path)

    def evaluate(self, eval_dataloader, steps=0):
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            for i,batch_data in enumerate(eval_dataloader):
                batch_data = self.strategy.to_device(batch_data)
                input_ids = batch_data["input_ids"]
                attention_mask = batch_data["attention_mask"]
                prompt_lengths = batch_data["prompt_lengths"]
                # forward
                output = self.model(input_ids,attention_mask=attention_mask)
                output.logits = output.logits.to(torch.float32)
                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    input_ids,
                    self.loss_fn.IGNORE_INDEX,
                )
                for label, source_len in zip(labels, prompt_lengths):
                    label[:source_len] = self.loss_fn.IGNORE_INDEX

                loss = self.loss_fn(output.logits, labels)

                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval gpt_loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if PrintUtil.is_rank_0():
                if self._swanlab is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._swanlab.log(logs)
        self.model.train()  # reset model state
