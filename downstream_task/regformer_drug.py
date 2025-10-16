#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/25 15:12
# @Author  : Luni Hu
# @File    : regformer_drug.py
# @Software: PyCharm

import os
import gc
import csv
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn.functional as F

import anndata
from tqdm import tqdm
import regformer as scmb
from regformer.utils.utils import load_config
from regformer.repo.deepcdr.drug_data_process import DrugDataProcess
from regformer.repo.deepcdr.drug import PyTorchMultiSourceGCNModel
from .regformer_emb import EmbTaskMamba
import swanlab


class DrugTaskMamba:
    """
    Class for predicting drug response (IC50) using cell line multi-omics features
    (gene expression, mutation, methylation) and drug molecular graph features.
    The gene expression feature is first processed into cell embeddings using
    a pre-trained Mamba-based model (EmbTaskMamba).
    """

    def __init__(self, config_file):
        """
        Initializes the drug response prediction task.
        :param config_file: Path to the configuration file.
        """
        self.args = load_config(config_file)  # Load configuration arguments
        self._init_logger()  # Set up the save directory and logger
        self.device = self.args.device  # Set compute device
        self.leave_drug = self.args.leave_drug  # Drug ID for leave-drug-out test
        self.embedder = EmbTaskMamba(config_file)  # Initialize the embedding task for feature extraction

    def _init_logger(self):
        """
        Creates the run directory, copies the script, and sets up file logging.
        """
        self.save_dir = Path(self.args.save_dir) / self.args.run_name  # Construct the full save path
        self.save_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        self.args.save_dir = self.save_dir  # Update args with the full save path
        print(f"Save to {self.save_dir}")
        os.system(f"cp {__file__} {self.save_dir}")  # Copy the script for record-keeping
        self.logger = scmb.logger  # Get the global logger instance
        scmb.utils.add_file_handler(self.logger, self.save_dir / "run.log")  # Add file output to logger

    def pretrain_inference(self, gexpr_feature):
        """
        Converts raw gene expression features into cell embeddings using the pre-trained Mamba model.
        :param gexpr_feature: Pandas DataFrame of raw gene expression, indexed by cell line ID.
        :return: pd.DataFrame: Normalized cell embeddings, indexed by cell line ID.
        """
        # Create an AnnData object temporarily for use with the embedding task's data loader logic
        adata = anndata.AnnData(X=gexpr_feature)
        adata.obs[self.args.cell_type_column] = None  # Mock cell type column
        # Load the pre-trained model and inference DataLoader
        model, loader = self.embedder.load_data_and_model(adata)
        self.logger.info(f"Start embedding. Total samples: {len(loader.dataset)}")

        # Initialize array to store extracted cell embeddings
        embeddings = np.zeros((len(loader.dataset), self.args.layer_size), dtype=np.float32)
        model.eval()
        count = 0

        # Perform inference with no gradient and mixed precision (if enabled)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            for batch in tqdm(loader, desc="Embedding"):
                input_ids = batch["gene_ids"].to(self.args.device)
                input_vals = batch["values"].to(self.args.device)
                pad_mask = input_ids.eq(self.embedder.vocab[self.embedder.pad_token])
                sorted_layer_idx = batch.get("sorted_layer_idx")
                if sorted_layer_idx is not None:
                    sorted_layer_idx = sorted_layer_idx.to(self.args.device)

                # Encode the input sequences
                out = model._encode(
                    src=input_ids,
                    values=input_vals,
                    batch_labels=None,
                    src_key_padding_mask=pad_mask,
                    # Pass sorted layer index if graph sorting and layer embedding are enabled
                    sorted_layer_idx=sorted_layer_idx if (
                                getattr(self.args, "graph_sort", False) and getattr(self.args, "layer_emb",
                                                                                    False)) else None
                )
                # Extract cell-level embedding
                batch_emb = model._get_cell_emb_from_layer(out, input_vals, src_key_padding_mask=pad_mask).cpu().numpy()
                embeddings[count:count + len(batch_emb)] = batch_emb
                count += len(batch_emb)

        # L2-normalize the extracted embeddings
        gexpr_emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.logger.info(f"Embedding size: {gexpr_emb.shape}")
        # Return embeddings as a DataFrame, preserving cell line indices
        return pd.DataFrame(gexpr_emb, index=gexpr_feature.index)

    def train(self, model, dataloader, validation_data, optimizer):
        """
        Trains the PyTorchMultiSourceGCNModel for drug response prediction.
        :param model: The drug response prediction model.
        :param dataloader: Training DataLoader.
        :param validation_data: Validation features and labels for testing the best model.
        :param optimizer: PyTorch optimizer.
        """
        patience, best_pcc, best_epoch, counter = 10, -np.Inf, 0, 0  # Early stopping/best model tracking setup

        for epoch in range(self.args.epochs):
            model.train()
            loss_list = []
            pcc_list = []

            for ii, data_ in enumerate(dataloader):
                # Move data to the specified device
                data_ = [d.to(self.device) for d in data_]
                X_drug_feat, X_drug_adj, X_mut, X_gexpr, X_methy, Y = data_

                # Forward pass: predict drug response
                output = model(X_drug_feat, X_drug_adj, X_mut, X_gexpr, X_methy).squeeze(-1)
                # Calculate Mean Squared Error (MSE) loss
                loss = F.mse_loss(output, Y)
                # Calculate Pearson Correlation Coefficient (PCC)
                pcc = torch.corrcoef(torch.stack((output, Y)))[0, 1]

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
                pcc_list.append(pcc.item())

                if ii % 500 == 0:
                    self.logger.info(f"| epoch {epoch} | batch {ii} | loss {loss.item():.4f} | pcc {pcc:.4f}")

            epoch_loss = np.mean(loss_list)
            epoch_pcc = np.mean(pcc_list)
            torch.save(model.state_dict(), self.save_dir / f"{epoch}.pth")  # Save checkpoint for the current epoch
            self.logger.info(f"[Epoch {epoch}] Loss {epoch_loss:.4f} | PCC {epoch_pcc:.4f} ")

            # Update best model based on epoch PCC
            if epoch_pcc > best_pcc:
                best_pcc, best_epoch, counter = epoch_pcc, epoch, 0

        self.logger.info(f"Best epoch {best_epoch} | Best PCC {best_pcc:.4f}")
        # Load the best model state dictionary
        model.load_state_dict(torch.load(self.save_dir / f"{best_epoch}.pth"))
        # Record final test result using the best model
        self._record_test_result(model, validation_data)

    def test(self, model, validation_data):
        """
        Evaluates the model on the validation/test data.
        :param model: The drug response prediction model.
        :param validation_data: Test features and labels.
        :return: (loss, pcc, spearman_correlation)
        """
        model.eval()
        with torch.no_grad():
            val_X, val_Y = validation_data
            # Move test features and labels to device
            val_X = [d.to(self.device) for d in val_X]
            val_Y = val_Y.to(self.device)

            # Create DataLoader for batch inference
            test_ds = TensorDataset(*val_X)
            test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

            # Predict outputs in batches
            outputs = [model(*batch).squeeze(-1) for batch in test_loader]
            output_test = torch.cat(outputs, dim=0)

            # Calculate metrics
            loss = F.mse_loss(output_test, val_Y)
            pcc = torch.corrcoef(torch.stack((output_test, val_Y)))[0, 1]
            spearman, _ = spearmanr(output_test.cpu().numpy(), val_Y.cpu().numpy())

        return loss.item(), pcc.item(), spearman

    def _record_test_result(self, model, validation_data):
        """
        Tests the final model, logs metrics, and saves results to a CSV file.
        :param model: The final trained/loaded model.
        :param validation_data: Test features and labels.
        """
        loss, pcc, spearman = self.test(model, validation_data)
        # Determine the test mode for file naming
        mode = "leave" if getattr(self.args, "leave_drug_test", False) else "random"
        csv_path = self.save_dir / f"{mode}_test.csv"

        # Append results to the CSV file
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([round(loss, 4), round(pcc, 4), round(spearman, 4)])
        self.logger.info(f"Test result saved to {csv_path}")
        self.logger.info(f"Loss {loss:.4f} | PCC {pcc:.4f} | Spearman {spearman:.4f}")

    def run(self, gexpr_emb=None):
        """
        Main method to run the entire drug response prediction pipeline.
        :param gexpr_emb: Optional pre-calculated gene expression embeddings.
        """
        random.seed(0)  # Set random seed for reproducibility
        data_obj = DrugDataProcess(self.args)  # Initialize data processing utility

        # Generate and preprocess raw multi-omics and drug features
        mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx = data_obj.MetadataGenerate(
            self.args.drug_info_file, self.args.cell_line_info_file, self.args.genomic_mutation_file,
            self.args.drug_feature_file, self.args.gene_expression_file, self.args.methylation_file)

        # Extract cell embeddings if not provided
        if gexpr_emb is None:
            gexpr_emb = self.pretrain_inference(gexpr_feature)

        # Split data into training and testing sets based on the specified mode
        if self.args.leave_drug_test:
            all_drug_ids = sorted(set(drug_feature.keys()))
            data_train_idx, data_test_idx = data_obj.DrugSplit(data_idx, all_drug_ids,
                                                               self.leave_drug)  # Leave-drug-out split
        else:
            data_train_idx, data_test_idx = data_obj.DataSplit(data_idx)  # Random split

        # Extract features for training and testing data points
        X_drug_data_train, X_mutation_data_train, X_gexpr_data_train, X_methylation_data_train, Y_train, cancer_type_train_list = data_obj.FeatureExtract(
            data_train_idx, drug_feature, mutation_feature, gexpr_emb, methylation_feature)
        X_drug_data_test, X_mutation_data_test, X_gexpr_data_test, X_methylation_data_test, Y_test, cancer_type_test_list = data_obj.FeatureExtract(
            data_test_idx, drug_feature, mutation_feature, gexpr_emb, methylation_feature)

        # Process drug features (molecule features and adjacency matrix) into Tensors
        X_drug_feat_data_train = torch.stack([item[0] for item in X_drug_data_train])
        X_drug_adj_data_train = torch.stack([item[1] for item in X_drug_data_train])
        X_drug_feat_data_test = torch.stack([item[0] for item in X_drug_data_test])
        X_drug_adj_data_test = torch.stack([item[1] for item in X_drug_data_test])

        # Prepare validation/test data structure
        validation_data = [[X_drug_feat_data_test, X_drug_adj_data_test, X_mutation_data_test, X_gexpr_data_test,
                            X_methylation_data_test], Y_test]

        # Save test data if not in leave-drug-out mode
        if not self.args.leave_drug_test:
            test_data = [[X_drug_feat_data_test, X_drug_adj_data_test, X_mutation_data_test, X_gexpr_data_test,
                          X_methylation_data_test], Y_test, data_test_idx]
            os.makedirs(f'{self.args.save_dir}/data/test_data/', exist_ok=True)
            torch.save(test_data, f'{self.args.save_dir}/data/test_data/test_data.pth')
            print(f'Test data saved to {self.args.save_dir}/data/test_data/test_data.pth')

        # Create training TensorDataset and DataLoader
        train_data = TensorDataset(X_drug_feat_data_train, X_drug_adj_data_train, X_mutation_data_train,
                                   X_gexpr_data_train, X_methylation_data_train, Y_train)
        dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

        # Initialize the Multi-Source GCN Model
        model = PyTorchMultiSourceGCNModel(
            drug_input_dim=X_drug_data_train[0][0].shape[-1], drug_hidden_dim=256, drug_concate_before_dim=100,
            mutation_input_dim=X_mutation_data_train.shape[-2], mutation_hidden_dim=256,
            mutation_concate_before_dim=100,
            gexpr_input_dim=X_gexpr_data_train.shape[-1], gexpr_hidden_dim=256, gexpr_concate_before_dim=100,
            methy_input_dim=X_methylation_data_train.shape[-1], methy_hidden_dim=256, methy_concate_before_dim=100,
            output_dim=300, units_list=self.args.unit_list, use_mut=self.args.use_mut, use_gexp=self.args.use_gexp,
            use_methy=self.args.use_methy, regr=True, use_relu=self.args.use_relu, use_bn=self.args.use_bn,
            use_GMP=self.args.use_GMP
        ).to(self.device)

        # Initialize the optimizer
        optimizer = Adam(model.parameters(), lr=self.args.lr, eps=1e-07)

        # Run training or testing based on configuration
        if self.args.mode == 'train':
            self.train(model, dataloader, validation_data, optimizer)
        elif self.args.mode == 'test':
            # Load pre-trained model for testing
            model_path = self.save_dir / f"best_{self.args.model_used}_model.pt"
            model.load_state_dict(torch.load(model_path))
            self._record_test_result(model, validation_data)

        gc.collect()  # Trigger garbage collection


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()

    task = DrugTaskMamba(args.config_file)
    task.run()