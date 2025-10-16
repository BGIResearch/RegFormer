#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/25
# @Author  : Luni Hu
# @File    : regformer_pert.py
# @Software: PyCharm

import os
import gc
import torch
import pickle
import shutil
import numpy as np
from pathlib import Path

import scanpy as sc
import swanlab
import regformer as scmb

from regformer.model.mambaLM import MambaModel
from regformer.repo.gears import PertData, GEARS
from regformer.repo.gears.inference import evaluate, compute_metrics

from regformer.utils.utils import (
    load_config, model_config, load_ckpt,
    gene2vec_embedding
)
from regformer.data.gene_tokenizer import GeneVocab

import warnings
warnings.filterwarnings("ignore")

class PertTaskMamba:
    """
    Class for the Perturbation Prediction task, utilizing gene embeddings
    derived from a pre-trained Mamba-based model (RegFormer/scMamba) within
    the GEARS framework.
    """
    def __init__(self, config_file):
        """
        Initializes the perturbation task.
        :param config_file: Path to the configuration file.
        """
        self.args = load_config(config_file)  # Load configuration arguments
        self.device = self.args.device  # Set compute device

        self._init_logger()  # Setup save directory and logger
        self._init_tokens(pad_token="<pad>", unk_token="<unk>")  # Initialize tokens and load pre-trained model parts

    def _init_logger(self):
        """
        Creates the run directory, copies the script, and sets up file logging.
        """
        self.save_dir = Path(self.args.save_dir) / self.args.run_name  # Construct the full save path
        self.save_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        print(f"Save to {self.save_dir}")
        os.system(f"cp {__file__} {self.save_dir}")  # Copy the script for record-keeping
        self.logger = scmb.logger  # Get the global logger instance
        scmb.utils.add_file_handler(self.logger, self.save_dir / "run.log")  # Add file output to logger

    def _init_tokens(self, pad_token, unk_token):
        """
        Defines special tokens and loads necessary components of the pre-trained Mamba model
        if a checkpoint is specified.
        """
        self.pad_token = pad_token
        # Determine the unk_token
        self.unk_token = unk_token if self.args.append_cls else "<cls>"
        self.mask_token = "<mask>"
        if self.args.input_emb_style == "category":
            # Numerical values for binned/categorical input
            self.mask_value = self.args.n_bins + 1
            self.pad_value = self.args.n_bins
            self.n_input_bins = self.args.n_bins + 2
        else:
            # Numerical values for continuous input
            self.mask_value, self.pad_value = -1, -2
            self.n_input_bins = self.args.n_bins

        self.vocab = None
        self.gene2ids = None
        self.load_obj = None

        if getattr(self.args, "load_model", None) is not None:
            # Load model components if pre-trained model is specified
            model_configs, vocab_file, model_file = model_config(self.args)
            self.vocab = GeneVocab.from_file(vocab_file)  # Load vocabulary

            # Add special tokens to the vocabulary
            for t in [self.pad_token, self.mask_token, self.unk_token]:
                if t not in self.vocab:
                    self.vocab.append_token(t)
            self.vocab.set_default_index(self.vocab[self.pad_token])

            self.load_obj = self._load_model(model_configs)  # Instantiate Mamba model
            self.load_obj = load_ckpt(self.load_obj, model_file, self.args, self.logger)  # Load checkpoint
            self.load_obj.to(self.device)  # Move model to device
            self.gene2ids = self.vocab.vocab.get_stoi()  # Get gene name to index mapping

    def _load_model(self, model_configs):
        """
        Instantiates the MambaModel primarily for the purpose of gene embedding extraction.
        :param model_configs: Dictionary containing general model parameters.
        :return: An instance of MambaModel.
        """
        return MambaModel(
            ntoken=len(self.vocab),
            d_model=model_configs['embsize'],
            nlayers=model_configs['nlayers'],
            device=self.device,
            vocab=self.vocab,
            dropout=self.args.dropout,
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            do_mvc=self.args.MVC,
            do_dab=False,
            domain_spec_batchnorm=False,
            n_input_bins=self.n_input_bins,
            input_emb_style=self.args.input_emb_style,
            cell_emb_style=self.args.cell_emb_style,
            pre_norm=self.args.pre_norm,
            do_pretrain=True,  # Set to True for embedding extraction
            topo_graph=self.args.graph_sort,
            if_bimamba=self.args.bimamba_type != "none",
            bimamba_type=self.args.bimamba_type,
            token_emb_freeze=getattr(self.args, "token_emb_freeze", False),
            only_value_emb=getattr(self.args, "only_value_emb", False),
            bin_cls=getattr(self.args, "bin_cls", False),
            bin_nums=self.args.n_bins,
            use_transformer=getattr(self.args, "use_transformer", False),
        )

    def make_pert_data(self):
        """
        Loads and prepares the perturbation data using the PertData class from GEARS.
        Subsets genes based on the Mamba model's vocabulary if a pre-trained model is used.
        :return: PertData object containing processed data, splits, and DataLoaders.
        """
        pert_data = PertData(self.args.data_path)

        # Determine gene subset based on vocabulary if pre-trained model is loaded
        gene_subset = (
            list(self.gene2ids.keys())
            if self.gene2ids
            else None
        )

        if self.args.data_name in ['norman', 'adamson']:
            # Load built-in GEARS datasets
            pert_data.load(data_name=self.args.data_name, gene_subset=gene_subset)
        else:
            # Load custom h5ad dataset
            adata = sc.read_h5ad(
                Path(self.args.data_dir) / f"{self.args.data_name}.h5ad"
            )
            sc.pp.calculate_qc_metrics(adata, inplace=True)
            pert_data.new_data_process(
                dataset_name=self.args.data_name, adata=adata
            )

        # Prepare gene-level train/test/ood splits
        pert_data.prepare_split(
            split=self.args.split,
            seed=self.args.seed,
            train_gene_set_size=self.args.train_gene_set_size
        )
        # Create DataLoaders for the splits
        pert_data.get_dataloader(
            batch_size=self.args.batch_size,
            test_batch_size=self.args.test_batch_size
        )

        return pert_data

    def get_gene_embedding(self, gene_ids):
        """
        Extracts gene embeddings from the pre-trained Mamba model's encoder.
        :param gene_ids: Tensor of gene indices.
        :return: Numpy array of gene embeddings.
        """
        self.logger.info('start to get gene embedding!')
        gene_ids = gene_ids.to(self.device)
        self.load_obj.eval()  # Set model to evaluation mode
        # The Mamba model's encoder (token embedding layer) is used here
        gene_embeddings = self.load_obj.encoder(gene_ids)
        self.logger.info('finished get gene embedding!')
        return gene_embeddings.detach().cpu().numpy()

    def universal_gene_embedding(self, pert_data):
        """
        Creates the gene embedding weight matrix for the GEARS model initialization.
        This matrix is initialized either randomly, with Mamba embeddings, or with gene2vec embeddings.
        :param pert_data: The PertData object.
        :return: Tensor of gene embedding weights (already on device).
        """
        gene_list = pert_data.gene_names.values.tolist()  # All genes in the perturbation data
        self.logger.info(f"Number of genes: {len(gene_list)}")

        # Initialize the embedding matrix randomly (PyTorch default)
        gene_emb_weight = torch.nn.Embedding(
            len(gene_list), self.args.layer_size
        ).weight.detach().to(self.device)

        if self.args.use_pretrained:
            if self.load_obj is not None:
                # Use gene embeddings from the pre-trained Mamba model
                gene_in_vocab = [
                    g for g in pert_data.adata.var.gene_name if g in self.gene2ids
                ]
                gene_ids = np.array([self.gene2ids[g] for g in gene_in_vocab])
                gene_ids = torch.tensor(gene_ids, dtype=torch.long).to(self.device)

                ids_embedding = self.get_gene_embedding(gene_ids)  # Extract embeddings

                # Fill the GEARS embedding matrix with the Mamba embeddings for overlapping genes
                for i, gene in enumerate(gene_in_vocab):
                    gene_emb_weight[gene_list.index(gene)] = torch.tensor(
                        ids_embedding[i], device=self.device
                    )
            elif self.args.model_used == 'gene2vec':
                # Use gene2vec embeddings if specified
                g2v_emb_dict = gene2vec_embedding(
                    self.args.g2v_file, self.args.g2v_genes
                )
                # Fill the GEARS embedding matrix with gene2vec embeddings for overlapping genes
                for gene, emb in g2v_emb_dict.items():
                    if gene in gene_list:
                        gene_emb_weight[gene_list.index(gene)] = torch.tensor(
                            emb, device=self.device
                        )

        return gene_emb_weight  # Return the initialized/populated embedding matrix

    def evaluate_pert(self, pert_data, model, output_file=None):
        """
        Evaluates the trained GEARS model on the test splits.
        :param pert_data: The PertData object.
        :param model: The GEARS model wrapper object.
        :param output_file: Path to save the evaluation results pickle file.
        :return: (test_res, test_metrics, test_pert_res)
        """
        self.logger.info(f"Test subgroups: {pert_data.subgroup['test_subgroup'].keys()}")
        self.logger.info(f"Conditions: {pert_data.adata.obs.condition.unique()}")

        model.best_model.to(self.device)  # Ensure the best model is on the correct device

        # Perform prediction and uncertainty estimation on the test set
        test_res = evaluate(
            model.dataloader['test_loader'],
            model.best_model,
            model.config['uncertainty'],
            self.device
        )
        # Compute evaluation metrics
        test_metrics, test_pert_res = compute_metrics(test_res)

        self.logger.info(f"Test metrics: {test_metrics}")

        if output_file is not None:
            output_file = Path(output_file)
            # Save evaluation results to a pickle file
            with output_file.open("wb") as f:
                pickle.dump([test_res, test_metrics, test_pert_res], f)
            self.logger.info(f"Saved evaluation results to {output_file}")

        return test_res, test_metrics, test_pert_res

    def run_pert_analysis(self, gene_emb_weight=None):
        """
        Main method to run the perturbation prediction pipeline: data setup, model initialization, and training.
        :param gene_emb_weight: Optional pre-computed gene embedding weights.
        :return: (pert_data, model)
        """
        self.logger.info(self.args)

        pert_data = self.make_pert_data()  # Prepare perturbation data
        self.logger.info(f"adata shape: {pert_data.adata.shape}")

        # Set maximum sequence length for the GEARS model based on gene count
        self.args.max_seq_len = pert_data.adata.shape[1]

        # Get gene embeddings if not provided
        if gene_emb_weight is None:
            gene_emb_weight = self.universal_gene_embedding(pert_data)

        # Initialize the GEARS model wrapper
        model = GEARS(
            pert_data,
            device=self.device,
            model_output=str(self.save_dir),
        )

        # Initialize the internal prediction model with gene embeddings
        model.model_initialize(
            hidden_size=self.args.hidden_size,
            use_pretrained=self.args.use_pretrained,
            pretrain_freeze=self.args.pretrain_freeze,
            gene_emb_weight=gene_emb_weight,
            pretrained_emb_size=self.args.layer_size,
            gene2ids=self.gene2ids,
            layer_size=self.args.layer_size
        )

        model.best_model.to(self.device)  # Ensure model is on device

        if self.args.finetune:
            # Run finetuning if configured
            model.train(
                epochs=self.args.epochs,
                lr=self.args.lr
            )
            model.save_model(str(self.save_dir))  # Save the trained model

        self.logger.info("Perturbation modeling complete.")
        gc.collect()

        return pert_data, model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()

    task = PertTaskMamba(args.config_file)
    pert_data, model = task.run_pert_analysis()
    # Evaluate the model and save results
    task.evaluate_pert(pert_data, model, output_file=task.save_dir / "result.pkl")