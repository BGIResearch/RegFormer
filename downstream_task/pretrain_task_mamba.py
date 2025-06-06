# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:pretrain_task_mamba.py
# @Software:PyCharm
# @Created Time:2024/6/1 3:48 PM
import os, torch
import gc
from torch import nn
import copy
import warnings
from regformer.utils.utils import load_config
from pathlib import Path
import wandb
import time
import torch.distributed as dist
import regformer as scmb
from regformer.utils.utils import seed_all, model_config, load_ckpt, define_wandb_metrcis
from regformer.data.dataset import Load_Data
from torch.utils.data import DataLoader
from regformer.data.dataloader import Get_DataLoader
from regformer.model.mambaLM import MambaModel
from regformer.utils.utils import get_reduced
from regformer.data.gene_tokenizer import GeneVocab
from regformer.model.loss import masked_mse_loss, masked_relative_error, masked_cross_entry_loss
import datetime
import shutil
from torch.nn.parallel import DistributedDataParallel as DDP

warnings.filterwarnings('ignore')


class PretrainTaskScMamba(object):
    def __init__(self, config_file, pad_token="<pad>", unk_token='<unk>'):
        self.args = load_config(config_file)
        if self.args.distributed:
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl')
            self.world_size = torch.distributed.get_world_size()
            self.device = torch.device("cuda", self.local_rank)
            seed_all(self.args.seed + torch.distributed.get_rank())
        else:
            os.environ["WANDB_MODE"] = "offline"
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.device = torch.device("cuda", self.local_rank)
        self.is_master = self.rank == 0

        save_dir = os.path.join(self.args.save_dir, self.args.task, self.args.data_name, self.args.model_name,
                                self.args.run_name)

        if self.is_master:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
            print(f"save to {self.save_dir}")
            # save the whole script to the dir
            os.system(f"cp {__file__} {self.save_dir}")
            self.logger = scmb.logger
            scmb.utils.add_file_handler(self.logger, self.save_dir / "run.log")
        else:
            self.logger = None
        seed_all(self.args.seed)
        #
        if self.args.input_emb_style == "category":
            self.mask_value = self.args.n_bins + 1
            self.pad_value = self.args.n_bins  # for padding gene expr values
            self.n_input_bins = self.args.n_bins + 2
        else:
            self.mask_value = -1
            self.pad_value = -2
            self.n_input_bins = self.args.n_bins
        self.pad_token, self.unk_token = pad_token, unk_token
        self.pad_token = "<pad>"
        self.unk_token = '<unk>' if self.args.append_cls else '<cls>'

    def load_data_and_model(self):
        # load config
        model_configs, vocab_file, model_file = model_config(self.args)
        vocab = GeneVocab.from_file(vocab_file)
        self.mask_token = '<mask>' if '<mask>' in vocab.vocab.itos_ else '<eoc>'

        special_tokens = [self.pad_token, self.mask_token, self.unk_token]
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        vocab.set_default_index(vocab["<pad>"])
        if self.is_master:
            shutil.copy(vocab_file, self.save_dir / "vocab.json")
        self.vocab = vocab

        # load data
        train_dataset, valid_dataset = self.load_data(vocab, pad_token=self.pad_token,
                                                      pad_value=self.pad_value, mask_value=self.mask_value)
        train_loader = Get_DataLoader(dataset=train_dataset, args=self.args, shuffle=False,
                                      drop_last=False, pin_memory=True)
        valid_loader = Get_DataLoader(dataset=valid_dataset, args=self.args, shuffle=False,
                                      drop_last=False, pin_memory=True)
        if self.is_master:
            self.logger.info(
                f"Total set number of cells: {len(train_dataset) + len(valid_dataset)}."
                f"\n\t Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")
            if not self.args.include_zero_gene:
                self.logger.info(f"\n\t Only using non-zero genes:"
                                 f"\n\tUniform the length into max_seq_len: {self.args.max_seq_len}")
            else:
                self.logger.info(
                    f"\n\t Using all the genes, the length of whole gene, uniform the length into max_seq_len: {self.args.max_seq_len}")

        # load model and ckpt
        model = self.load_model(model_configs, vocab)
        if self.args.load_model:
            model = load_ckpt(model, model_file, self.args, self.logger)
        model = model.to(self.device)
        return model, train_loader, valid_loader

    def load_data(self, vocab, pad_token="<pad>", pad_value=-2, mask_value=-1):
        '''
        Loading LMDB dataset
        Args:
            vocab:
            pad_token:
            pad_value:
            mask_value:

        Returns:

        '''
        data_path = os.path.join(self.args.data_path, self.args.task, self.args.data_name)
        train_dataset, valid_dataset = Load_Data(data_path=data_path, args=self.args,
                                                 vocab=vocab, mask_ratio=self.args.mask_ratio,
                                                 append_cls=self.args.append_cls,
                                                 include_zero_gene=self.args.include_zero_gene,
                                                 max_seq_len=self.args.max_seq_len,
                                                 mask_token=self.mask_token, unk_token=self.unk_token)
        # data_configs={'num_batch_types':num_batch_types,'celltypes':celltypes,'id2type':id2type,
        #              'num_types':num_types,'adata_test_raw':adata_test_raw,'test_labels':test_data_pt['celltype_labels']}
        # self.cls_count = torch.bincount(train_data_pt['celltype_labels'])
        return train_dataset, valid_dataset

    def load_model(self, model_configs, vocab):
        args = self.args
        ntokens = len(vocab)
        only_value_emb = self.args.only_value_emb if 'only_value_emb' in self.args else False
        bin_cls = self.args.bin_cls if 'bin_cls' in self.args else False
        use_transformer = self.args.use_transformer if 'use_transformer' in self.args else False
        model = MambaModel(
            ntoken=ntokens, d_model=model_configs['embsize'], nlayers=model_configs['nlayers'],
            vocab=vocab, dropout=args.dropout, pad_token=self.pad_token, pad_value=self.pad_value, do_mvc=args.MVC,
            do_dab=False, domain_spec_batchnorm=self.args.DSBN,
            n_input_bins=self.n_input_bins, input_emb_style=args.input_emb_style,
            cell_emb_style=args.cell_emb_style, mvc_decoder_style=args.mvc_decoder_style, pre_norm=args.pre_norm,
            do_pretrain=True, topo_graph=args.graph_sort, if_bimamba=args.bimamba_type != "none",
            bimamba_type=args.bimamba_type,
            if_devide_out=False, init_layer_scale=None, token_emb_freeze=args.token_emb_freeze,
            only_value_emb=only_value_emb, bin_cls=bin_cls, bin_nums=self.args.n_bins, use_transformer=use_transformer)
        return model

    def set_wandb(self):
        ## wandb setting
        now = datetime.datetime.now().strftime("%Y-%m-%d")
        wandb_name = f'{self.args.data_name}_{self.args.model_name}{str(self.args.bimamba_type) if self.args.bimamba_type != "none" else ""}_{"lye" if self.args.layer_emb else ""}_{"CLM" if self.args.generative_pretraining else "MLM"}_{self.args.run_name}_G{self.world_size}_{now}'
        wandb_tags = [self.args.task, f'G{self.world_size}',
                      self.args.data_name, self.args.model_name, 'gts' if self.args.graph_sort else 'wogts',
                      f'layer_mask{self.args.mask_ratio}' if self.args.layer_mask else f'random_mask{self.args.mask_ratio}',
                      'layer_positional_emb' if self.args.layer_emb else 'w/o lyemb']
        self.run = wandb.init(
            config=self.args.__dict__,
            job_type=self.args.task,
            project="scLLM-Pretrain",
            name=wandb_name,
            tags=wandb_tags,
            reinit=True,
            settings=wandb.Settings(start_method="fork"),
        )
        print(self.args)

    def load_criterion_and_opt(self, model):
        self.criterion = masked_cross_entry_loss if "bin_cls"  in self.args and self.args.bin_cls else masked_mse_loss
        if self.args.graph_sort:
            self.lm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.args.lr, eps=1e-4 if self.args.amp else 1e-8
        )
        schedule_interval = max(1, int(self.args.epochs * 0.1))
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, schedule_interval, gamma=self.args.schedule_ratio
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.amp)

    def train(self, model, loader, epoch):
        model.train()
        total_loss, total_mse, total_MVC, total_topo = 0.0, 0.0, 0.0, 0.0
        total_error = 0.0
        log_interval = self.args.log_interval
        start_time = time.time()
        num_batches = len(loader)
        for batch_idx, batch_data in enumerate(loader):
            model.zero_grad()
            input_gene_ids = batch_data["gene_ids"].to(self.device)
            src_key_padding_mask = input_gene_ids.eq(self.vocab[self.pad_token])
            input_values = batch_data["masked_values"].to(self.device)  # masked-> -1
            target_values = batch_data["target_values"].to(self.device)
            if self.args.graph_sort:
                target_sorted_gene = batch_data['sorted_gene_ids'].to(self.device)
                input_sorted_gene = batch_data['masked_sorted_gene_ids'].to(self.device)
                sorted_layer_idx = batch_data['sorted_layer_idx'].to(self.device)
                topo_padding_mask = input_sorted_gene.eq(self.vocab[self.pad_token]).to(self.device)
            else:
                input_sorted_gene = None
                topo_padding_mask = None
                target_sorted_gene = None
            with torch.cuda.amp.autocast(enabled=self.args.amp):
                output_dict = model(
                    src=input_gene_ids,
                    values=input_values,
                    batch_labels=None,
                    MVC=self.args.MVC,
                    src_key_padding_mask=src_key_padding_mask,
                    input_sorted_gene=input_sorted_gene if not self.args.generative_pretraining else target_sorted_gene,
                    topo_padding_mask=topo_padding_mask,
                    sorted_layer_idx=sorted_layer_idx if (self.args.graph_sort and self.args.layer_emb) else None
                )
                ## masked_value_prediction
                masked_positions = input_values.eq(self.mask_value)
                loss = loss_mse = self.criterion(
                    output_dict["mlm_output"], target_values, masked_positions
                )
                metrics_to_log = {"train/mlm": loss_mse.item()}

                if torch.isnan(loss):
                    self.logger.warning(f"[NaN Skip] Epoch {epoch}, Batch {batch_idx}: loss_mse is NaN.")
                    continue
                if self.args.graph_sort:
                    if self.args.generative_pretraining:
                        logit = output_dict['lm_logit'][:, :-1, :].clone()
                        logit = logit.view(-1, output_dict['lm_logit'].size(-1))
                        label = target_sorted_gene[:, 1:].clone()
                        label = label.view(-1).long()
                        padded_positions = label.eq(self.vocab[self.pad_token])
                        label[padded_positions] = -100
                    else:  # randolym masked token prediction
                        logit = output_dict['lm_logit'].view(-1, output_dict['lm_logit'].size(-1))
                        label = target_sorted_gene.view(-1).long()
                        topo_needed_to_pred = input_sorted_gene.eq(self.vocab[self.mask_token]).to(self.device)
                        masked_pos = torch.logical_or(topo_padding_mask, ~topo_needed_to_pred).view(-1)
                        label[masked_pos] = -100
                    topo_sorting_loss = self.lm_criterion(logit, label)
                    if torch.isnan(topo_sorting_loss) and self.is_master:
                        self.logger.warning(f"[NaN Skip] Epoch {epoch}, Batch {batch_idx}: topo_sorting_loss is NaN.")
                        continue
                    weight = loss_mse.item() / (topo_sorting_loss.item() + 1e-8)
                    loss = loss + weight * topo_sorting_loss
                    metrics_to_log.update({"train/topo_loss": topo_sorting_loss.item()})
                if self.args.MVC:
                    loss_MVC = self.criterion(
                        output_dict["mvc_output"], target_values, masked_positions
                    )
                    if torch.isnan(loss_MVC)  and self.is_master:
                        self.logger.warning(f"[NaN Skip] Epoch {epoch}, Batch {batch_idx}: loss_MVC is NaN.")
                        continue
                    weight = loss_mse.item() / (loss_MVC.item() + 1e-8)
                    loss = loss + weight * loss_MVC
                    metrics_to_log.update({"train/mvc": loss_MVC.item()})
            if torch.isnan(loss) and self.is_master:
                self.logger.warning(f"[NaN Skip] Epoch {epoch}, Batch {batch_idx}: final loss is NaN.")
                continue
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if self.scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    if self.is_master:
                        self.logger.warning(
                            f"Found infinite gradient. This may be caused by the gradient "
                            f"scaler. The current scale is {self.scaler.get_scale()}. This warning "
                            "can be ignored if no longer occurs after autoscaling of the scaler."
                        )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.is_master:
                wandb.log(metrics_to_log)
            with torch.no_grad():
                mre = masked_relative_error(
                    output_dict["mlm_output"], target_values, masked_positions
                ) if ('bin_cls' in self.args and not self.args.bin_cls) or 'bin_cls' not in self.args else -1
            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_MVC += loss_MVC.item() if self.args.MVC else 0.0
            total_topo += topo_sorting_loss.item() if self.args.graph_sort else 0.0
            total_error += mre.item() if ('bin_cls' in self.args and not self.args.bin_cls) or 'bin_cls' not in self.args else 0
            if batch_idx % log_interval == 0 and batch_idx > 0:
                lr = self.scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                cur_mse = total_mse / log_interval
                cur_gepc = total_MVC / log_interval if self.args.MVC else 0.0
                cur_topo = total_topo / log_interval if self.args.graph_sort else 0.0
                cur_error = total_error / log_interval
                # ppl = math.exp(cur_loss)
                if self.args.distributed:
                    cur_loss = get_reduced(cur_loss, self.local_rank, 0, self.world_size)
                    cur_mse = get_reduced(cur_mse, self.local_rank, 0, self.world_size)
                    cur_gepc = get_reduced(cur_gepc, self.local_rank, 0, self.world_size)
                    cur_error = get_reduced(cur_error, self.local_rank, 0, self.world_size)
                    cur_topo = get_reduced(cur_topo, self.local_rank, 0, self.world_size)
                if self.is_master:
                    self.logger.info(
                        f"| epoch {epoch:3d} | {batch_idx:3d}/{num_batches:3d} batches | "
                        f"lr {lr:05.6f} | ms/batch {ms_per_batch:5.2f} | "
                        f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                        + (f"gepc {cur_gepc:5.2f} |" if self.args.MVC else "")
                        + (f"topo {cur_topo:5.2f} |" if self.args.graph_sort else "")
                    )
                total_loss = 0
                total_mse = 0
                total_gepc = 0
                total_error = 0
                total_topo = 0
                start_time = time.time()

    def evaluate(self, model: nn.Module, loader: DataLoader, epoch) -> float:
        """
        Evaluate the model on the evaluation data.
        """
        model.eval()
        # if self.args.distributed:
        #     dist.barrier()
        total_loss = 0.0
        total_error = 0.0
        total_num = 0
        total_topo = 0
        total_mvc = 0
        with torch.no_grad():
            for batch_data in loader:
                input_gene_ids = batch_data["gene_ids"].to(self.device)
                src_key_padding_mask = input_gene_ids.eq(self.vocab[self.pad_token])
                input_values = batch_data["masked_values"].to(self.device)  # masked-> -1
                target_values = batch_data["target_values"].to(self.device)
                if self.args.graph_sort:
                    target_sorted_gene = batch_data['sorted_gene_ids'].to(self.device)
                    input_sorted_gene = batch_data['masked_sorted_gene_ids'].to(self.device)
                    sorted_layer_idx = batch_data['sorted_layer_idx'].to(self.device)
                    topo_padding_mask = input_sorted_gene.eq(self.vocab[self.pad_token]).to(self.device)
                else:
                    input_sorted_gene = None
                    topo_padding_mask = None
                    sorted_layer_idx = None
                    target_sorted_gene = None
                with torch.cuda.amp.autocast(enabled=self.args.amp):
                    output_dict = model(
                        src=input_gene_ids,
                        values=input_values,
                        MVC=self.args.MVC,
                        batch_labels=None,
                        src_key_padding_mask=src_key_padding_mask,
                        input_sorted_gene=input_sorted_gene if not self.args.generative_pretraining else target_sorted_gene,
                        topo_padding_mask=topo_padding_mask,
                        sorted_layer_idx=sorted_layer_idx if (self.args.graph_sort and self.args.layer_emb) else None
                    )
                    # causal loss
                    output_values = output_dict["mlm_output"]
                    masked_positions = input_values.eq(self.mask_value)
                    loss = self.criterion(output_values, target_values, masked_positions)
                    if self.args.graph_sort:
                        if self.args.generative_pretraining:
                            logit = output_dict['lm_logit'][:, :-1, :].clone()
                            logit = logit.view(-1, output_dict['lm_logit'].size(-1))
                            label = target_sorted_gene[:, 1:].clone()
                            label = label.view(-1).long()
                            padded_positions = label.eq(self.vocab[self.pad_token])
                            label[padded_positions] = -100
                        else:  # randolym masked token prediction, MLM TASK
                            logit = output_dict['lm_logit'].view(-1, output_dict['lm_logit'].size(-1))
                            label = target_sorted_gene.view(-1).long()
                            topo_needed_to_pred = input_sorted_gene.eq(self.vocab[self.mask_token]).to(self.device)
                            masked_pos = torch.logical_or(topo_padding_mask, ~topo_needed_to_pred).view(-1)
                            label[masked_pos] = -100
                        topo_sorting_loss = self.lm_criterion(logit, label)
                        weight = loss.item() / topo_sorting_loss.item()
                        loss = loss + weight * topo_sorting_loss
                        # loss = loss + topo_sorting_loss
                    if self.args.MVC:
                        loss_gepc = self.criterion(
                            output_dict["mvc_output"], target_values, masked_positions
                        )
                        weight = loss.item() / loss_gepc.item()
                        loss = loss + weight * loss_gepc
                total_loss += loss.item() * len(input_gene_ids)
                total_topo += topo_sorting_loss.item() * len(input_gene_ids) if self.args.graph_sort else 0
                total_mvc += loss_gepc.item() * len(input_gene_ids) if self.args.MVC else 0
                total_error += masked_relative_error(
                    output_values, target_values, masked_positions
                ).item() * len(input_gene_ids) if ('bin_cls' in self.args and not self.args.bin_cls) or 'bin_cls' not in self.args else 0
                # total_dab += loss_dab.item() * len(input_gene_ids)
                total_num += len(input_gene_ids)
        if self.args.distributed:
            mse = get_reduced(total_loss / total_num, self.local_rank, 0, self.world_size)
            mre = get_reduced(total_error / total_num, self.local_rank, 0, self.world_size)
            topo_mse = get_reduced(total_topo / total_num, self.local_rank, 0, self.world_size)
            mvc_mse = get_reduced(total_mvc / total_num, self.local_rank, 0, self.world_size)
        else:
            mse = total_loss / total_num
            mre = total_error / total_num
            topo_mse = total_topo / total_num
            mvc_mse = total_mvc / total_num
        total_mse = mse + topo_mse + mvc_mse
        if self.is_master:
            wandb.log(
                {
                    "valid/mse": mse,
                    "valid/mre": mre,
                    "valid/topo": topo_mse,
                    "valid/mvc": mvc_mse,
                    # / total_num,
                    "epoch": epoch,
                })

        return mse, mre, total_mse

    def run_pretrain(self, ):
        if self.is_master:
            self.set_wandb()
            define_wandb_metrcis()
        model, train_loader, valid_loader = self.load_data_and_model()
        if self.is_master:
            from regformer.utils.utils import print_model_size
            print_model_size(model)
        if self.args.distributed:
            model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank,
                find_unused_parameters=True)
        self.load_criterion_and_opt(model)
        if self.is_master:
            wandb.watch(model, log='parameters')
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"Parameter {name} has no grad")

        best_val_loss = float("inf")
        best_model = copy.deepcopy(model)
        best_model_epoch = 1
        args = self.args

        for epoch in range(1, args.epochs + 1):
            # if args.distributed:
            #     dist.barrier()
            epoch_start_time = time.time()
            if args.do_train:
                self.train(
                    model,
                    loader=train_loader,
                    epoch=epoch
                )
                val_loss, val_mre, val_total_mse = self.evaluate(
                    model,
                    loader=valid_loader,
                    epoch=epoch
                )
                elapsed = time.time() - epoch_start_time
                if self.is_master:
                    self.logger.info("-" * 89)
                    self.logger.info(
                        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                        f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
                    )
                    self.logger.info("-" * 89)
                if val_total_mse < best_val_loss:
                    best_val_loss = val_total_mse
                    best_model = copy.deepcopy(model)
                    best_model_epoch = epoch
                    if self.is_master:
                        self.logger.info(f"Best model with total mse score {best_val_loss:5.4f}")
                if epoch == 1 or epoch % args.save_eval_interval == 0 or epoch == args.epochs:
                    if self.is_master:
                        self.logger.info(f"Saving model to {self.save_dir}")
                        if args.distributed:
                            torch.save(best_model.module.state_dict(), self.save_dir / f"model_e{best_model_epoch}.pt")
                        else:
                            torch.save(best_model.state_dict(), self.save_dir / f"model_e{best_model_epoch}.pt")
            # if self.args.distributed:
            #     dist.barrier()
            self.scheduler.step()
        if self.is_master:
            if args.distributed:
                torch.save(best_model.module.state_dict(), self.save_dir / "best_model.pt")
            else:
                torch.save(best_model.state_dict(), self.save_dir / "best_model.pt")
            artifact = wandb.Artifact(f"best_model", type="model")
            glob_str = os.path.join(self.save_dir, "best_model.pt")
            artifact.add_file(glob_str)
            self.run.log_artifact(artifact)

            self.run.finish()
            wandb.finish()
            gc.collect()


if __name__ == "__main__":
    import sys

    debug = False
    if debug:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    config_file = sys.argv[1]
    task = PretrainTaskScMamba(config_file)
    task.run_pretrain()
