#!/usr/bin/env python3
# coding: utf-8
"""
@author: Qiuping
@file: cell_emb_regformer
@time: 2025/5/20 17:19

Change Log：
2025/5/20 17:19: create file
"""

import os
import gc
import torch
import shutil
import swanlab
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import silhouette_score

import regformer as scmb
from regformer.utils.utils import (
    load_config, model_config, load_ckpt
)
from regformer.model.mambaLM import MambaModel
from regformer.data.dataset import infer_dataset, SeqDataset
from regformer.data.gene_tokenizer import GeneVocab
from regformer.utils.utils import refine_embedding

class EmbTaskMamba:
    """
    Class for extracting cell embeddings from a pre-trained Mamba-based model (RegFormer/scMamba).
    Handles data loading, model inference, embedding refinement, and result visualization/evaluation.
    """
    def __init__(self, config_file, pad_token="<pad>", unk_token="<unk>"):
        """
        Initializes the embedding task with configuration, logger, and tokens.
        :param config_file: Path to the configuration file.
        :param pad_token: Token used for padding sequences.
        :param unk_token: Token used for unknown genes or CLS token.
        """
        self.args = load_config(config_file)  # Load run configuration arguments
        self._init_logger()  # Setup save directory and logger
        self._init_tokens(pad_token, unk_token)  # Define special tokens and their numerical values
        self._check_parameters()  # Validate the configuration settings

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
        Defines the names and numerical values for special tokens based on input embedding style.
        """
        self.pad_token = pad_token
        # Determine the unk_token; it acts as <cls> if CLS token is not appended
        self.unk_token = unk_token if self.args.append_cls else "<cls>"
        self.mask_token = "<mask>"
        if self.args.input_emb_style == "category":
            # Numerical values for binned/categorical input
            self.mask_value = self.args.n_bins + 1  # Mask index
            self.pad_value = self.args.n_bins  # Pad index
            self.n_input_bins = self.args.n_bins + 2  # Total categories: bins + pad + mask
        else:
            # Numerical values for continuous input
            self.mask_value, self.pad_value = -1, -2
            self.n_input_bins = self.args.n_bins

    def _check_parameters(self):
        """
        Validates the compatibility of input/output style and embedding style parameters.
        """
        # Assertions for supported styles
        assert self.args.input_style in ["normed_raw", "log1p", "binned"]
        assert self.args.output_style in ["normed_raw", "log1p", "binned"]
        assert self.args.input_emb_style in ["category", "continuous", "scaling"]
        # Check for invalid combination: binned input with scaling embedding
        if self.args.input_style == "binned" and self.args.input_emb_style == "scaling":
            raise ValueError("input_emb_style 'scaling' not supported for binned input.")
        # Check for invalid combination: continuous input with category embedding
        if self.args.input_style in ["log1p", "normed_raw"] and self.args.input_emb_style == "category":
            raise ValueError("input_emb_style 'category' not supported for log1p/normed_raw input.")

    def load_data_and_model(self, adata=None):
        """
        Loads model configuration, vocabulary, dataset, DataLoader, and the Mamba model.
        :param adata: Optional anndata object to load data from (instead of data_path).
        :return: model (MambaModel on device), data_loader (DataLoader for inference)
        """
        model_configs, vocab_file, model_file = model_config(self.args)  # Get model config and file paths
        self.vocab = GeneVocab.from_file(vocab_file)  # Load the gene vocabulary

        # Add special tokens to the vocabulary if missing
        for t in [self.pad_token, self.mask_token, self.unk_token]:
            if t not in self.vocab:
                self.vocab.append_token(t)
        self.vocab.set_default_index(self.vocab[self.pad_token])  # Set padding index as the default

        shutil.copy(vocab_file, self.save_dir / "vocab.json")  # Copy vocabulary file to run directory

        # Load and preprocess data for inference
        data_pt = infer_dataset(
            data_path=self.args.data_path,
            adata=adata,
            args=self.args,
            logger=self.logger,
            vocab=self.vocab,
            is_master=True,
            mask_value=self.mask_value,
            pad_value=self.pad_value,
            pad_token=self.pad_token
        )
        # Check if classification is intended and calculate class counts
        if getattr(self.args, "CLS", False):
            self.cls_count = torch.bincount(data_pt['celltype_labels'])
        dataset = SeqDataset(data_pt)  # Create a sequence dataset
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)  # Create DataLoader for inference

        model = self._load_model(model_configs)  # Instantiate the Mamba model
        if self.args.load_model is not None:
            model = load_ckpt(model, model_file, self.args, self.logger)  # Load pre-trained checkpoint

        return model.to(self.args.device), data_loader

    def _load_model(self, model_configs):
        """
        Instantiates the MambaModel with specific configurations for embedding extraction.
        :param model_configs: Dictionary containing general model parameters.
        :return: An instance of MambaModel.
        """
        return MambaModel(
            ntoken=len(self.vocab),
            d_model=model_configs['embsize'],
            nlayers=model_configs['nlayers'],
            device=self.args.device,
            vocab=self.vocab,
            dropout=self.args.dropout,
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            do_mvc=self.args.MVC,
            do_dab=False,
            domain_spec_batchnorm=self.args.DSBN,  # Domain-specific Batch Normalization flag
            num_batch_labels=getattr(self.args, "num_batch_labels", None),  # Number of batch labels for DSBN
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

    def run_embedding(self):
        """
        Executes the embedding process: loads model/data, performs inference, saves embeddings,
        and organizes results (including plotting UMAP and calculating metrics).
        """
        model, loader = self.load_data_and_model()  # Load model and data
        self.logger.info(f"Start embedding. Total samples: {len(loader.dataset)}")

        # Initialize array to store extracted cell embeddings
        embeddings = np.zeros((len(loader.dataset), self.args.layer_size), dtype=np.float32)
        all_batch_labels = []
        all_celltype_labels = []

        model.eval()  # Set model to evaluation mode
        count = 0

        # Perform inference with no gradient and mixed precision (if enabled)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            for batch in tqdm(loader, desc="Embedding"):
                input_ids = batch["gene_ids"].to(self.args.device)
                input_vals = batch["values"].to(self.args.device)
                batch_ids = batch["batch_labels"].to(self.args.device)

                sorted_layer_idx = batch.get("sorted_layer_idx")
                if sorted_layer_idx is not None:
                    sorted_layer_idx = sorted_layer_idx.to(self.args.device)
                pad_mask = input_ids.eq(self.vocab[self.pad_token])  # Padding mask

                # Encode the input sequences to get hidden states
                out = model._encode(
                    src=input_ids,
                    values=input_vals,
                    batch_labels= batch_ids if self.args.DSBN else None,  # Pass batch labels for DSBN
                    src_key_padding_mask=pad_mask,
                    # Pass sorted layer index if graph sorting and layer embedding are enabled
                    sorted_layer_idx=sorted_layer_idx if (self.args.graph_sort and self.args.layer_emb) else None
                )

                # Extract cell-level embedding (e.g., from CLS token or sequence pooling)
                batch_emb = model._get_cell_emb_from_layer(out, input_vals, src_key_padding_mask=pad_mask).cpu().numpy()
                all_batch_labels.append(batch_ids.cpu().numpy())
                all_celltype_labels.append(batch["celltype_labels"].cpu().numpy())

                # Store embeddings and update counter
                embeddings[count:count + len(batch_emb)] = batch_emb
                count += len(batch_emb)

        # Concatenate and print unique labels for verification
        batch_labels = np.concatenate(all_batch_labels)
        print(np.unique(batch_labels))
        celltype_labels = np.concatenate(all_celltype_labels)
        print(np.unique(celltype_labels))

        # Refine embeddings using techniques like batch correction (if needed)
        embeddings = refine_embedding(embeddings, batch_labels, celltype_labels, 2)
        np.save(self.save_dir / "cell_embedding.npy", embeddings)  # Save the final embeddings

        # Organize results: update anndata, plot UMAP, calculate metrics
        self._organize_results(sc.read_h5ad(self.args.data_path), embeddings)
        self.logger.info("Embedding complete.")
        gc.collect()

    def _organize_results(self, adata, embeddings):
        """
        Adds embeddings to anndata, optionally generates UMAP visualization, and calculates the AWS score.
        :param adata: The anndata object (raw data).
        :param embeddings: The extracted cell embeddings.
        """
        adata.obsm['emb'] = embeddings  # Store embeddings in anndata.obsm

        # Generate UMAP visualization if enabled
        if getattr(self.args, "draw_umap", False):
            sc.pp.neighbors(adata, use_rep='emb')  # Compute nearest neighbors using embeddings
            sc.tl.umap(adata)  # Compute UMAP coordinates
            with plt.rc_context({"figure.figsize": (8, 4), "figure.dpi": (300)}):
                sc.pl.umap(adata, color=self.args.cell_type_column, show=False)  # Plot UMAP colored by cell type
                plt.title(self.args.run_name)
                plt.savefig(self.save_dir / "cell_emb_umap.png", dpi=300)

            # Log UMAP image to swanlab
            swanlab.log({
                "test/cell_emb_umap": swanlab.Image(str(self.save_dir / "cell_emb_umap.png"), caption="UMAP"),
            })

        # Calculate Adjusted Within-cluster Silhouette (AWS) score (using cosine metric)
        labels = adata.obs[self.args.cell_type_column].values
        # Formula: (1 + cosine_silhouette_score) / 2
        score = (1 + silhouette_score(embeddings, labels, metric="cosine")) / 2
        self.logger.info(f"AWS score: {score:.4f}")
        # swanlab.log({"test/aws": score}) # Log AWS score to swanlab


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()
    task = EmbTaskMamba(args.config_file)
    task.run_embedding()