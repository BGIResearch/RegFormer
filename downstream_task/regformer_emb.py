#!/usr/bin/env python3
# coding: utf-8
"""
@author: Qiuping
@file: cell_emb_regformer
@time: 2025/5/20 17:19

Change Log：
2025/5/20 17:19: create file
"""
import os, torch
from regformer.utils.utils import load_config
from pathlib import Path
import regformer as scmb
from regformer.utils.utils import seed_all, model_config, load_ckpt
from regformer.data.dataset import infer_dataset, SeqDataset
from torch.utils.data import DataLoader
from regformer.model.mambaLM import MambaModel
from regformer.data.gene_tokenizer import GeneVocab
import shutil
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import wandb
import scanpy as sc
import gc
from sklearn.metrics import silhouette_score


class RegformerEmb(object):
    def __init__(self, config_file, pad_token="<pad>", unk_token='<unk>'):

        self.args = load_config(config_file)
        self.device = self.args.device
        self.check_parameters()
        save_dir = f'{self.args.save_dir}/{self.args.run_name}'
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"save to {self.save_dir}")
        # save the whole script to the dir
        os.system(f"cp {__file__} {self.save_dir}")
        self.logger = scmb.logger
        scmb.utils.add_file_handler(self.logger, self.save_dir / "run.log")
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

    def check_parameters(self):
        assert self.args.input_style in ["normed_raw", "log1p", "binned"]
        assert self.args.output_style in ["normed_raw", "log1p", "binned"]
        assert self.args.input_emb_style in ["category", "continuous", "scaling"]
        if self.args.input_style == "binned":
            if self.args.input_emb_style == "scaling":
                raise ValueError("input_emb_style `scaling` is not supported for binned input.")
        elif self.args.input_style == "log1p" or self.args.input_style == "normed_raw":
            if self.args.input_emb_style == "category":
                raise ValueError(
                    "input_emb_style `category` is not supported for log1p or normed_raw input."
                )

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
        shutil.copy(vocab_file, self.save_dir / "vocab.json")
        self.vocab = vocab
        # load data
        train_data_pt = self.load_data(vocab, pad_token=self.pad_token,
                                       pad_value=self.pad_value, mask_value=self.mask_value)
        dataset = SeqDataset(train_data_pt)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
        )
        # load model and ckpt
        model = self.load_model(model_configs, vocab)
        if self.args.load_model is not None:
            model = load_ckpt(model, model_file, self.args, self.logger)
        model = model.to(self.device)
        return model, data_loader

    def load_data(self, vocab, pad_token="<pad>", pad_value=-2, mask_value=-1):
        train_data_pt = infer_dataset(data_path=self.args.data_path, args=self.args, logger=self.logger, vocab=vocab,
                                      is_master=True,
                                      mask_value=mask_value, pad_value=pad_value, pad_token=pad_token)
        self.cls_count = torch.bincount(train_data_pt['celltype_labels'])
        return train_data_pt

    def load_model(self, model_configs, vocab):
        args = self.args
        ntokens = len(vocab)
        num_batch_types = 0
        # model = MambaModel(ntoken=ntokens, d_model=model_configs['embsize'], nlayers=model_configs['nlayers'],
        #                    nlayers_cls=3, n_cls=1,
        #                    device=self.device, vocab=vocab, dropout=args.dropout, pad_token=self.pad_token,
        #                    pad_value=self.pad_value, num_batch_labels=num_batch_types,
        #                    n_input_bins=self.n_input_bins, input_emb_style=args.input_emb_style,
        #                    cell_emb_style=args.cell_emb_style,
        #                    pre_norm=args.pre_norm, do_pretrain=False, if_bimamba=args.bimamba_type != "none",
        #                    bimamba_type=args.bimamba_type, if_devide_out=False, init_layer_scale=None)
        only_value_emb = self.args.only_value_emb if 'only_value_emb' in self.args else False
        bin_cls = self.args.bin_cls if 'bin_cls' in self.args else False
        use_transformer = self.args.use_transformer if 'use_transformer' in self.args else False
        model = MambaModel(
            ntoken=ntokens, d_model=model_configs['embsize'], nlayers=model_configs['nlayers'],
            nlayers_cls=3, n_cls=1,
            device=self.device, vocab=vocab, dropout=args.dropout, pad_token=self.pad_token,
            pad_value=self.pad_value, do_mvc=False,
            do_dab=False, domain_spec_batchnorm=False,
            n_input_bins=self.n_input_bins, input_emb_style=args.input_emb_style,
            cell_emb_style=args.cell_emb_style, pre_norm=args.pre_norm,
            do_pretrain=False, topo_graph=args.graph_sort, if_bimamba=args.bimamba_type != "none",
            bimamba_type=args.bimamba_type,
            if_devide_out=False, init_layer_scale=None, token_emb_freeze=args.token_emb_freeze,
            only_value_emb=only_value_emb, bin_cls=bin_cls, bin_nums=self.args.n_bins, use_transformer=use_transformer)
        return model

    def set_wandb(self):
        wandb_name = f'{self.args.run_name}'
        wandb_tags = ['cellemb']
        self.run = wandb.init(
            config=self.args.__dict__,
            project="Regformer_CellEmb",
            name=wandb_name,
            tags=wandb_tags,
            reinit=True,
            settings=wandb.Settings(start_method="fork"),
        )
        print(self.args.__dict__)

    def get_cell_embedding(self):
        self.set_wandb()
        model, data_loader = self.load_data_and_model()
        model = model.to(self.device)
        self.logger.info(f'start to get cell embedding! sample number: {len(data_loader.dataset)}')
        cell_embeddings = np.zeros((len(data_loader.dataset), self.args.layer_size), dtype=np.float32)
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            count = 0
            for batch_data in tqdm(data_loader, desc='Cell embedding'):
                input_gene_ids = batch_data["gene_ids"].to(self.args.device)
                input_values = batch_data["values"].to(self.args.device)
                if self.args.graph_sort and self.args.layer_emb:
                    sorted_layer_idx = batch_data['sorted_layer_idx'].to(self.args.device)
                else:
                    sorted_layer_idx = None
                src_key_padding_mask = input_gene_ids.eq(self.vocab[self.pad_token])
                output = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    sorted_layer_idx=sorted_layer_idx
                )
                embeddings = output['cell_emb']  # get the <cls> position embedding
                embeddings = embeddings.cpu().numpy()
                cell_embeddings[count: count + len(embeddings)] = embeddings
                count += len(embeddings)
        self.logger.info(f"cell embedding shape: {cell_embeddings.shape}")
        np.save(f'{self.save_dir}/cell_embedding.npy', cell_embeddings)
        adata = sc.read_h5ad(self.args.data_path)
        self.result_organizing(adata, cell_embeddings)
        self.logger.info("get cell embedding Done!")
        self.run.finish()
        wandb.finish()
        gc.collect()

    def result_organizing(self, adata, cell_embeddings):
        with plt.rc_context({"figure.figsize": (8, 4), "figure.dpi": (300)}):
            adata.obsm['emb'] = cell_embeddings
            sc.pp.neighbors(adata, use_rep='emb')
            sc.tl.umap(adata)
            sc.pl.umap(adata, color=self.args.cell_type_column, show=False)
            plt.savefig(self.save_dir / "cell_emb_umap.png", dpi=300)
        results = {}
        labels = adata.obs[self.args.cell_type_column].values
        score = (1+silhouette_score(cell_embeddings, labels))/2
        self.logger.info(f"aws: {score}")
        results["test/cell_emb_umap"] = wandb.Image(
            str(self.save_dir / "cell_emb_umap.png"),
            caption=f"cell_emb_umap",
        )
        results['test/aws'] = score
        wandb.log(results)


if __name__ == "__main__":
    # config_file = sys.argv[1]
    config_files = [
        r'/home/share/huadjyin/home/s_qiuping1/workspace/RegFormer/example/conf/cell_emb/graph_sort_1.toml',
        r'/home/share/huadjyin/home/s_qiuping1/workspace/RegFormer/example/conf/cell_emb/random_graph_sort_1.toml',
        r'/home/share/huadjyin/home/s_qiuping1/workspace/RegFormer/example/conf/cell_emb/mlm_1.toml',
        r'/home/share/huadjyin/home/s_qiuping1/workspace/RegFormer/example/conf/cell_emb/all_length_1.toml',
        r'/home/share/huadjyin/home/s_qiuping1/workspace/RegFormer/example/conf/cell_emb/shuffled_graph_1.toml',
        r'/home/share/huadjyin/home/s_qiuping1/workspace/RegFormer/example/conf/cell_emb/removed_graph_1.toml',
        r'/home/share/huadjyin/home/s_qiuping1/workspace/RegFormer/example/conf/cell_emb/transformer_1.toml',
    ]
    for config_file in config_files:
        task = RegformerEmb(config_file)
        task.get_cell_embedding()