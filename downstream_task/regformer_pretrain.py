#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/16 11:04
# @Author  : Luni Hu
# @File    : regformer_pretrain.py
# @Software: PyCharm

import os, torch, datetime, warnings, copy, shutil, gc, time
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR, StepLR

import swanlab
import regformer as scmb
from regformer.utils.utils import (
    seed_all, model_config, load_ckpt, get_reduced, load_config
)
from regformer.data.dataset import Load_Data
from regformer.data.dataloader import Get_DataLoader
from regformer.model.mambaLM import MambaModel
from regformer.model.loss import masked_mse_loss, masked_cross_entry_loss, MultiTaskLoss
from regformer.data.gene_tokenizer import GeneVocab

warnings.filterwarnings('ignore')


class PretrainTaskScMamba:
    """
    Orchestrates the pre-training of a Mamba-based foundation model for single-cell genomics.
    This class manages distributed training, data loading, model instantiation, and the execution
    of self-supervised learning objectives.
    """

    def __init__(self, config_file, pad_token="<pad>", unk_token='<unk>'):
        """Initializes the pre-training task, setting up distributed environment and configuration."""
        self.args = load_config(config_file)
        self._init_distributed()
        self._init_logger()
        self._init_tokens(pad_token, unk_token)

    def _init_distributed(self):
        """Initializes the distributed training environment using PyTorch DistributedDataParallel (DDP)."""
        if self.args.distributed:
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl')
        else:
            self.rank = self.local_rank = 0
        self.device = torch.device("cuda", self.local_rank)
        self.is_master = self.rank == 0  # Master process handles logging and saving.
        self.world_size = dist.get_world_size() if self.args.distributed else 1
        seed_all(self.args.seed + self.rank)

    def _init_logger(self):
        """Initializes the output directory and logging for experimental reproducibility."""
        self.save_dir = Path(self.args.save_dir) / self.args.run_name
        self.resume_dir = self.save_dir.with_name(self.args.run_name + "_resume")
        os.makedirs(self.resume_dir, exist_ok=True)
        if self.is_master:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Save to {self.save_dir}")
            os.system(f"cp {__file__} {self.save_dir}")  # Save a code snapshot.
            self.logger = scmb.logger
            scmb.utils.add_file_handler(self.logger, self.save_dir / "run.log")

    def _init_tokens(self, pad_token, unk_token):
        """Defines special tokens and their numerical representations for sequence modeling."""
        self.pad_token, self.mask_token = pad_token, '<mask>'
        self.unk_token = unk_token if self.args.append_cls else '<cls>'
        if self.args.input_emb_style == "category":  # For binned expression values
            self.mask_value, self.pad_value = self.args.n_bins + 1, self.args.n_bins
            self.n_input_bins = self.args.n_bins + 2
        else:  # For continuous expression values
            self.mask_value, self.pad_value = -1, -2
            self.n_input_bins = self.args.n_bins

    def load_data_and_model(self):
        """
        Prepares datasets and initializes the Mamba model. This includes loading the gene vocabulary,
        creating data loaders with masked inputs for self-supervision, and instantiating the model
        architecture, potentially loading from a checkpoint.
        """
        model_configs, vocab_file, model_file = model_config(self.args)
        self.vocab = GeneVocab.from_file(vocab_file)
        for t in [self.pad_token, self.mask_token, self.unk_token]:
            if t not in self.vocab: self.vocab.append_token(t)
        self.vocab.set_default_index(self.vocab[self.pad_token])
        if self.is_master: shutil.copy(vocab_file, self.save_dir / "vocab.json")

        # Load data, which is pre-processed for self-supervised tasks (e.g., masking).
        train_set, val_set = Load_Data(data_path=self.args.data_path, args=self.args, vocab=self.vocab)
        if self.is_master: self.logger.info(f"Train samples: {len(train_set)}, Valid samples: {len(val_set)}")

        train_loader = Get_DataLoader(train_set, self.args, shuffle=True)
        valid_loader = Get_DataLoader(val_set, self.args, shuffle=False)

        model = self.load_model(model_configs)
        if self.args.load_model: model = load_ckpt(model, model_file, self.args, self.logger)
        if self.is_master:
            self.logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model.to(self.device), train_loader, valid_loader

    def load_model(self, model_configs):
        """Instantiates the MambaModel architecture with specified hyperparameters."""
        return MambaModel(
            ntoken=len(self.vocab),
            d_model=model_configs['embsize'],
            nlayers=model_configs['nlayers'],
            # (additional model parameters from config)
            do_pretrain=True
        )

    def load_criterion_and_opt(self, model):
        """
        Sets up the loss functions, optimizer, and learning rate scheduler.
        The loss is a composite of multiple self-supervised objectives.
        """
        # Define loss functions for different pre-training tasks.
        self.criterion = masked_cross_entry_loss if self.args.bin_cls else masked_mse_loss  # For Masked Language/Value Modeling.
        self.lm_criterion = nn.CrossEntropyLoss(
            ignore_index=-100) if self.args.graph_sort else None  # For topological (autoregressive) objective.

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, eps=1e-4 if self.args.amp else 1e-8)

        # Configure a learning rate scheduler with a linear warmup phase to stabilize initial training.
        if self.args.warmup_steps > 0:
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=self._build_warmup_lambda(self.args.warmup_steps,
                                                                                          self.args.warmup_start_lr / self.args.lr))
        else:
            self.scheduler = StepLR(self.optimizer, step_size=max(1, int(self.args.epochs * 0.1)),
                                    gamma=self.args.schedule_ratio)

        # Use a gradient scaler for stable and efficient training with automatic mixed precision (AMP).
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.amp)

    def _build_warmup_lambda(self, warmup_steps: int, init_scale: float):
        """Builds a lambda function for linear learning rate warmup."""

        def _lr_lambda(step: int):
            if step < warmup_steps: return init_scale + (1.0 - init_scale) * (step / max(1, warmup_steps))
            return 1.0

        return _lr_lambda

    def set_swanlab(self):
        """Initializes an experiment tracking session with SwanLab."""
        name = f'{self.args.data_name}_{self.args.model_name}_{self.args.run_name}_G{self.world_size}'
        self.run = swanlab.init(project="scLLM-Pretrain", experiment_name=name, config=self.args.__dict__)

    def _sanitize_loss(self, loss: torch.Tensor, name: str, epoch: int, batch_idx: int) -> torch.Tensor:
        """Checks for and handles non-finite (NaN or Inf) loss values to prevent training divergence."""
        has_nan = torch.isnan(loss).any()
        has_inf = torch.isinf(loss).any()
        if self.args.distributed:
            # Synchronize status across all processes.
            flags = torch.tensor([float(has_nan), float(has_inf)], device=loss.device)
            dist.all_reduce(flags, op=dist.ReduceOp.SUM)
            has_nan, has_inf = flags[0] > 0, flags[1] > 0

        if has_nan or has_inf:
            if self.is_master: self.logger.warning(
                f"Non-finite loss '{name}' at epoch {epoch}, batch {batch_idx}. Replacing with 0.")
            return torch.tensor(0.0, device=loss.device, requires_grad=True)
        return loss

    def train(self, model, loader, epoch):
        """Executes a single training epoch with a multi-task self-supervised objective."""
        model.train()
        total_loss, total_mse, total_topo, total_mvc = 0., 0., 0., 0.
        start_time = time.time()

        for batch_idx, batch in enumerate(loader):
            # Load and prepare batch data.
            input_ids, input_vals = batch["gene_ids"].to(self.device), batch["masked_values"].to(self.device)
            target_vals, mask = batch["target_values"].to(self.device), input_ids.eq(self.vocab[self.pad_token])
            masked_pos = input_vals.eq(self.mask_value)

            sorted_target = batch['sorted_gene_ids'].to(self.device) if self.args.graph_sort else None
            # Forward pass with Automatic Mixed Precision.
            with torch.cuda.amp.autocast(enabled=self.args.amp):
                out = model(src=input_ids, values=input_vals, MVC=self.args.MVC, TOPO=self.args.TOPO,
                            src_key_padding_mask=mask)

            # Compute individual loss components for the multi-task objective.
            mlm_loss = self.criterion(out["mlm_output"], target_vals, masked_pos) if self.args.MLM else None
            mvc_loss = masked_mse_loss(out["mvc_output"], target_vals, masked_pos) if self.args.MVC else None
            if self.args.TOPO:
                label = sorted_target[:, 1:].reshape(
                    -1).long() if self.args.generative_pretraining else sorted_target.reshape(-1).long()
                logit = out['lm_logit'][:, :-1, :].reshape(-1, out['lm_logit'].size(
                    -1)) if self.args.generative_pretraining else out['lm_logit'].reshape(-1, out['lm_logit'].size(-1))
                topo_loss = self.lm_criterion(logit, label)
            else:
                topo_loss = None

            # Combine losses, weighted by their respective alpha coefficients.
            final_loss = sum(
                (comp * alpha if comp is not None else 0.0)
                for comp, alpha in
                [(mlm_loss, self.args.alpha_mlm), (topo_loss, self.args.alpha_topo), (mvc_loss, self.args.alpha_mvc)]
            )
            final_loss = self._sanitize_loss(final_loss, "final_loss", epoch, batch_idx)

            # Standard backpropagation steps using the gradient scaler.
            self.scaler.scale(final_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            model.zero_grad()

            # Aggregate and log metrics.
            with torch.no_grad():
                total_loss += get_reduced(final_loss.item(), self.device, 0, self.world_size)
                if self.is_master and batch_idx % self.args.log_interval == 0 and batch_idx > 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    self.logger.info(
                        f"| epoch {epoch:3d} | batch {batch_idx:5d}/{len(loader)} | lr {self.optimizer.param_groups[0]['lr']:.6f} | loss {avg_loss:.4f}")
                    swanlab.log({"train/loss": avg_loss, "epoch": epoch})

    def evaluate(self, model, loader, epoch):
        """Evaluates the model on the validation set to monitor performance and prevent overfitting."""
        model.eval()
        loss_list, mse_list, topo_list, mvc_list = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                # Evaluation logic mirrors the training forward pass without backpropagation.
                input_ids, input_vals = batch["gene_ids"].to(self.device), batch["masked_values"].to(self.device)
                target_vals, mask = batch["target_values"].to(self.device), input_ids.eq(self.vocab[self.pad_token])
                masked_pos = input_vals.eq(self.mask_value)
                sorted_target = batch['sorted_gene_ids'].to(self.device) if self.args.graph_sort else None

                with torch.cuda.amp.autocast(enabled=self.args.amp):
                    out = model(src=input_ids, values=input_vals, MVC=self.args.MVC, TOPO=self.args.TOPO,
                                src_key_padding_mask=mask)

                mlm_loss = self.criterion(out["mlm_output"], target_vals,
                                          masked_pos) if self.args.MLM else torch.tensor(0.)
                mvc_loss = masked_mse_loss(out["mvc_output"], target_vals,
                                           masked_pos) if self.args.MVC else torch.tensor(0.)
                if self.args.TOPO:
                    label = sorted_target[:, 1:].reshape(
                        -1).long() if self.args.generative_pretraining else sorted_target.reshape(-1).long()
                    logit = out['lm_logit'][:, :-1, :].reshape(-1, out['lm_logit'].size(
                        -1)) if self.args.generative_pretraining else out['lm_logit'].reshape(-1,
                                                                                              out['lm_logit'].size(-1))
                    topo_loss = self.lm_criterion(logit, label)
                else:
                    topo_loss = torch.tensor(0.)

                final_loss = mlm_loss * self.args.alpha_mlm + topo_loss * self.args.alpha_topo + mvc_loss * self.args.alpha_mvc
                loss_list.append(final_loss.item())

        # Reduce metrics across all distributed processes for a global average.
        avg_total = get_reduced(np.mean(loss_list), self.device, 0, self.world_size)

        if self.is_master:
            self.logger.info(f"[Epoch {epoch}] valid_loss: {avg_total:.4f}")
            swanlab.log({"valid/loss": avg_total, "epoch": epoch})
        return {"total": avg_total}

    def run_pretrain(self):
        """Executes the complete pre-training and validation pipeline over multiple epochs."""
        if self.is_master: self.set_swanlab()
        model, train_loader, val_loader = self.load_data_and_model()

        if self.args.distributed:
            model = DDP(model, device_ids=[self.local_rank], find_unused_parameters=True)
        self.load_criterion_and_opt(model)
        best_val_loss = float("inf")

        for epoch in range(1, self.args.epochs + 1):
            self.train(model, train_loader, epoch)
            val_losses = self.evaluate(model, val_loader, epoch)
            self.scheduler.step()

            if self.is_master:
                # Save the model checkpoint if it achieves a new best validation loss.
                if val_losses["total"] < best_val_loss:
                    best_val_loss = val_losses["total"]
                    best_model = copy.deepcopy(model.module if self.args.distributed else model)
                    save_path = self.save_dir / "best_model.pt"
                    torch.save(best_model.state_dict(), save_path)
                    self.logger.info(f"Best model saved to {save_path} with validation loss {best_val_loss:.4f}")
            gc.collect()
            torch.cuda.empty_cache()

        if self.is_master: swanlab.finish()


if __name__ == "__main__":
    import sys

    config_file = sys.argv[1]
    task = PretrainTaskScMamba(config_file)
    task.run_pretrain()