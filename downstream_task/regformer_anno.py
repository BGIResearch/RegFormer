# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:anno_task_scmamba.py
# @Software:PyCharm
# @Created Time:2024/5/23 4:58 PM

import os, gc, shutil, copy, time, pickle, warnings
import torch
import numpy as np
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import regformer as scmb
from regformer.utils.utils import (
    load_config, seed_all, model_config, load_ckpt
)
from regformer.model.mambaLM import MambaModel
from regformer.data.gene_tokenizer import GeneVocab
from regformer.data.dataset import Load_Data, SeqDataset
from regformer.data.dataloader import Get_DataLoader


class AnnoTaskMamba:
    """
    Main class for the single-cell cell type annotation task using a Mamba-based model (scMamba).
    Handles initialization, data loading, model setup, training, validation, and testing.
    """
    def __init__(self, config_file, pad_token="<pad>", unk_token="<unk>"):
        """
        Initializes the task, loads configuration, sets device, logger, tokens, and seeds.
        :param config_file: Path to the configuration file (YAML/JSON).
        :param pad_token: Token used for sequence padding.
        :param unk_token: Token used for unknown genes or the CLS token based on settings.
        """
        self.args = load_config(config_file)  # Load run configuration arguments
        self.device = self.args.device  # Set compute device (e.g., 'cuda' or 'cpu')

        self._initLogger()  # Setup save directory and logger
        self._init_tokens(pad_token, unk_token)  # Define special tokens and their numerical values
        self._checkParameters()  # Validate input/output/embedding style compatibility

        seed_all(self.args.seed)  # Set random seeds for reproducibility

    def _initLogger(self):
        """
        Creates the run directory, copies the script, and sets up file logging.
        """
        self.save_dir = Path(self.args.save_dir) / self.args.run_name  # Construct the full save path
        self.save_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        print(f"Save to {self.save_dir}")
        os.system(f"cp {__file__} {self.save_dir}")  # Copy the script for documentation

        self.logger = scmb.logger  # Get the global logger instance
        scmb.utils.add_file_handler(self.logger, self.save_dir / "run.log")  # Add file output to logger

    def _init_tokens(self, pad_token, unk_token):
        """
        Determines the special tokens and their numerical indices/values based on the input embedding style.
        """
        self.pad_token = pad_token
        # Unk token may act as <cls> token if it's not explicitly appended
        self.unk_token = unk_token if self.args.append_cls else "<cls>"
        self.mask_token = "<mask>"
        if self.args.input_emb_style == "category":
            # Numerical values for binned/categorical input
            self.mask_value = self.args.n_bins + 1  # Mask index
            self.pad_value = self.args.n_bins  # Pad index
            self.n_input_bins = self.args.n_bins + 2  # Total number of bins/categories (bins + pad + mask)
        else:
            # Numerical values for continuous input (typically negative)
            self.mask_value, self.pad_value = -1, -2
            self.n_input_bins = self.args.n_bins  # n_bins parameter is used, but not as category count

    def _checkParameters(self):
        """
        Validates the compatibility of configuration parameters for data styles.
        Raises ValueError for unsupported combinations.
        """
        # Assertions for supported input/output styles and embedding style
        assert self.args.input_style in ["normed_raw", "log1p", "binned"]
        assert self.args.output_style in ["normed_raw", "log1p", "binned"]
        assert self.args.input_emb_style in ["category", "continuous", "scaling"]
        # Check for invalid combination: binned input with scaling embedding
        if self.args.input_style == "binned" and self.args.input_emb_style == "scaling":
            raise ValueError("input_emb_style 'scaling' not supported for binned input.")
        # Check for invalid combination: continuous input with category embedding
        if self.args.input_style in ["log1p", "normed_raw"] and self.args.input_emb_style == "category":
            raise ValueError("input_emb_style 'category' not supported for log1p/normed_raw input.")


    def load_data_and_model(self):
        """
        Loads model configurations, vocabulary, datasets, data loaders, and the Mamba model.
        Loads a pre-trained checkpoint if specified.
        :return: model, train_loader, valid_loader, test_loader, data_configs
        """
        model_configs, vocab_file, model_file = model_config(self.args)  # Get model configuration and file paths
        self.vocab = GeneVocab.from_file(vocab_file)  # Load the gene vocabulary

        # Add special tokens to the vocabulary and set the default (pad) index
        for t in [self.pad_token, self.mask_token, self.unk_token]:
            if t not in self.vocab:
                self.vocab.append_token(t)
        self.vocab.set_default_index(self.vocab[self.pad_token])

        dst_file = self.save_dir / "vocab.json"
        if not dst_file.exists():
            shutil.copy(vocab_file, dst_file)  # Copy vocabulary file to the run directory

        # Load and preprocess data into PyTorch tensors (pt format)
        train_pt, valid_pt, test_pt, data_configs = self._load_data(
            vocab=self.vocab,
            mask_value=self.mask_value,
            pad_value=self.pad_value,
            pad_token=self.pad_token
        )

        self.cls_count = torch.bincount(train_pt["celltype_labels"])  # Count class samples for loss weighting

        # Initialize data loaders for train, validation, and test sets
        train_loader = Get_DataLoader(train_pt, args=self.args, intra_domain_shuffle=True, weighted=True,
                                      drop_last=False)
        valid_loader = Get_DataLoader(valid_pt, args=self.args, intra_domain_shuffle=False,
                                      drop_last=False)
        test_loader = Get_DataLoader(test_pt, args=self.args, intra_domain_shuffle=False,
                                      drop_last=False)

        model = self._load_model(model_configs, data_configs)  # Instantiate the Mamba model
        if not self.args.from_scratch:
            model = load_ckpt(model, model_file, self.args, self.logger)  # Load checkpoint if specified
        # model = self.freeze_model(model, unfreeze_modules=["mamba_encoder", "cls_decoder"]) if self.args.freeze else model  # Optional freezing

        return model.to(self.device), train_loader, valid_loader, test_loader, data_configs

    def _load_data(self, vocab, pad_token="<pad>", pad_value=-2, mask_value=-1):
        """
        Calls the utility function to load and preprocess data into train/valid/test tensors.
        Collects essential data configuration details.
        """
        train_data_pt, valid_data_pt, test_data_pt, num_batch_types, celltypes, id2type, num_types, adata_test_raw = \
            Load_Data(data_path=self.args.data_path, args=self.args, logger=self.logger, vocab=vocab, is_master=True,
                      mask_value=mask_value, pad_value=pad_value, pad_token=pad_token)
        # Consolidate data-related configurations
        data_configs = {'num_batch_types': num_batch_types, 'celltypes': celltypes, 'id2type': id2type,
                        'num_types': num_types, 'adata_test_raw': adata_test_raw,
                        'test_labels': test_data_pt['celltype_labels']}
        self.cls_count = torch.bincount(train_data_pt['celltype_labels'])  # Re-calculate class counts (redundant but safe)
        return train_data_pt, valid_data_pt, test_data_pt, data_configs

    def _load_model(self, model_configs, data_configs):
        """
        Instantiates the MambaModel with configurations from both model and data settings.
        :return: An instance of MambaModel.
        """
        return MambaModel(
            ntoken=len(self.vocab),
            d_model=model_configs["embsize"],
            nlayers=model_configs["nlayers"],
            nlayers_cls=self.args.nlayers_cls,
            n_cls=data_configs["num_types"] if self.args.CLS else 1,
            device=self.device,
            vocab=self.vocab,
            dropout=self.args.dropout,
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            do_mvc=False,
            do_dab=False,
            domain_spec_batchnorm=False,
            n_input_bins=self.n_input_bins,
            input_emb_style=self.args.input_emb_style,
            cell_emb_style=self.args.cell_emb_style,
            pre_norm=self.args.pre_norm,
            do_pretrain=self.args.do_pretrain,
            topo_graph=self.args.graph_sort,
            if_bimamba=self.args.bimamba_type != "none",
            bimamba_type=self.args.bimamba_type,
            token_emb_freeze=getattr(self.args, "token_emb_freeze", False),
            only_value_emb=getattr(self.args, "only_value_emb", False),
            bin_cls=getattr(self.args, "bin_cls", False),
            bin_nums=self.args.n_bins,
            use_transformer=getattr(self.args, "use_transformer", False)
        )

    def freeze_model(self, model, unfreeze_modules=None):
        """
        Freezes all parameters in the model, optionally unfreezing specified modules.
        :param model: The PyTorch model.
        :param unfreeze_modules: List of module names (or substrings) to unfreeze.
        :return: The modified model.
        """
        for param in model.parameters():
            param.requires_grad = False  # Freeze all parameters

        if unfreeze_modules is not None:
            for name, module in model.named_modules():
                if any(unfreeze in name for unfreeze in unfreeze_modules):
                    for param in module.parameters():
                        param.requires_grad = True  # Unfreeze specified modules

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable} / {total} ({trainable / total:.2%})")

        return model

    def _loadCriterionAndOpt(self, model):
        """
        Initializes the loss criterion (CrossEntropyLoss, potentially weighted) and the optimizer/scheduler.
        :param model: The PyTorch model.
        """
        if self.args.cls_weight:
            # Calculate inverse frequency weights for class imbalance
            cls_weight = self.cls_count.sum() / self.cls_count.float()
            # Handle potential Inf values (e.g., from zero counts) and normalize
            cls_weight = torch.where(torch.isinf(cls_weight), torch.tensor(0.0), cls_weight)
            cls_weight = cls_weight / cls_weight.sum()
            self.criterion_cls = nn.CrossEntropyLoss(weight=cls_weight.to(self.device))
        else:
            self.criterion_cls = nn.CrossEntropyLoss()  # Standard CrossEntropyLoss

        # Initialize Adam optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, eps=1e-4, betas=(0.9, 0.95), weight_decay=1e-5)

        # Initialize StepLR scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, max(10, int(self.args.epochs * 0.1)), gamma=self.args.schedule_ratio
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.amp)  # Initialize gradient scaler for AMP

    def _train(self, model, loader, epoch):
        """
        Performs a single training epoch over the DataLoader.
        :param model: The PyTorch model.
        :param loader: The training DataLoader.
        :param epoch: Current epoch number.
        """
        model.train()

        total_loss, total_cls, total_cce, total_err = 0.0, 0.0, 0.0, 0.0
        total_batches = len(loader)
        start_time = time.time()

        for batch_idx, batch in enumerate(loader):
            input_ids = batch["gene_ids"].to(self.device)
            input_vals = batch["values"].to(self.device)
            celltype_labels = batch["celltype_labels"].to(self.device)
            pad_mask = input_ids.eq(self.vocab[self.pad_token])  # Padding mask for attention

            sorted_layer_idx = batch.get("sorted_layer_idx")
            # If graph sorting and layer embedding are enabled, move layer index to device
            if self.args.graph_sort and self.args.layer_emb:
                sorted_layer_idx = sorted_layer_idx.to(self.device)
            else:
                sorted_layer_idx= None

            # Mixed precision training
            with torch.cuda.amp.autocast(enabled=self.args.amp, dtype=torch.bfloat16):
                output = model(
                    input_ids,
                    input_vals,
                    src_key_padding_mask=pad_mask,
                    CLS=self.args.CLS,  # Classification head flag
                    CCE=self.args.CCE,  # Continuous Cell Embedding flag
                    TOPO=self.args.TOPO, # Topological loss flag
                    sorted_layer_idx=sorted_layer_idx # Gene order information
                )

                loss = 0.0
                metrics = {}

                # Calculate Classification Loss (CLS)
                if self.args.CLS:
                    loss_cls = self.criterion_cls(output["cls_output"], celltype_labels)
                    loss += loss_cls
                    metrics["train/cls"] = loss_cls.item()
                    # Calculate classification error rate
                    error_rate = 1 - (output["cls_output"].argmax(1) == celltype_labels).sum().item() / len(input_ids)
                else:
                    error_rate = 0.0

                # Calculate Continuous Cell Embedding Loss (CCE)
                if self.args.CCE:
                    loss_cce = 10 * output["loss_cce"] # Weighted CCE loss
                    loss += loss_cce
                    metrics["train/cce"] = loss_cce.item()

            if torch.isnan(loss).any():
                self.logger.warning("NaN in loss — skipping batch.")
                self.scaler.update() # Update scaler without step, essentially skipping the batch
                continue

            metrics["train/loss"] = loss.item()

            # Backward pass and optimization using GradScaler
            model.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer) # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_cls += metrics.get("train/cls", 0.0)
            total_cce += metrics.get("train/cce", 0.0)
            total_err += error_rate

            # Log training progress at intervals
            if (batch_idx + 1) % self.args.log_interval == 0:
                lr = self.scheduler.get_last_lr()[0] if self.scheduler else 0.0
                elapsed = time.time() - start_time

                # Compute average metrics since last log
                avg_loss = total_loss / self.args.log_interval
                avg_cls = total_cls / self.args.log_interval if self.args.CLS else None
                avg_cce = total_cce / self.args.log_interval if self.args.CCE else None
                avg_err = total_err / self.args.log_interval if self.args.CLS else None

                log_msg = (
                    f"| epoch {epoch:3d} | batch {batch_idx + 1:5d}/{total_batches} "
                    f"| lr {lr:.6f} | loss {avg_loss:.4f} | {elapsed:.2f} sec "
                )
                if avg_cls is not None:
                    log_msg += f"| cls {avg_cls:.4f} "
                if avg_cce is not None:
                    log_msg += f"| cce {avg_cce:.4f} "
                if avg_err is not None:
                    log_msg += f"| err {avg_err:.4f}"

                self.logger.info(log_msg)

                # Reset counters for the next logging interval
                total_loss = 0.0
                total_cls = 0.0
                total_cce = 0.0
                total_err = 0.0

                start_time = time.time()

    def _evaluate(self, model, loader, epoch):
        """
        Performs validation/evaluation on a given DataLoader.
        :param model: The PyTorch model.
        :param loader: The validation DataLoader.
        :param epoch: Current epoch number.
        :return: (mean_squared_error_loss, error_rate, accuracy, precision, recall, f1_score)
        """
        check_model_weights(model)  # Check model weights integrity

        model.eval()
        total_loss = 0.0
        total_err = 0
        total_num = 0

        all_preds = []
        all_labels = []


        with torch.no_grad():
            for batch in loader:
                input_ids = batch["gene_ids"].to(self.device)
                input_vals = batch["values"].to(self.device)
                celltype_labels = batch["celltype_labels"].to(self.device)
                pad_mask = input_ids.eq(self.vocab[self.pad_token])

                sorted_layer_idx = batch.get("sorted_layer_idx")
                # Move sorted layer index to device if needed
                if self.args.graph_sort and self.args.layer_emb:
                    sorted_layer_idx = sorted_layer_idx.to(self.device)
                else:
                    sorted_layer_idx = None

                with torch.cuda.amp.autocast(enabled=False): # Disable AMP for evaluation
                    output = model(
                        input_ids,
                        input_vals,
                        src_key_padding_mask=pad_mask,
                        CLS=self.args.CLS,
                        CCE=self.args.CCE,
                        sorted_layer_idx=sorted_layer_idx
                    )
                    logits = output["cls_output"]

                    # Safety check for unstable logits
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print("⚠️ NaN/Inf detected in logits!")
                        print("  mean =", logits.mean().item())

                        raise ValueError("NaN/Inf in logits")

                    loss = self.criterion_cls(logits, celltype_labels)

                total_loss += loss.item()
                preds = torch.softmax(logits, dim=1).argmax(dim=1)
                total_err += (preds != celltype_labels).sum().item()
                total_num += input_ids.size(0)

                all_preds.append(preds.cpu())
                all_labels.append(celltype_labels.cpu())

        # Compute aggregate classification metrics
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        mse = total_loss / total_num  # Loss per sample (not strictly MSE, but average loss)
        err = total_err / total_num  # Error rate
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        self.logger.info(
            f"[Eval] epoch {epoch} | mse {mse:.4f} | err {err:.4f} | acc {acc:.4f} | precision {prec:.4f} | recall {rec:.4f} | f1 {f1:.4f}")


        return mse, err, acc, prec, rec, f1

    def _test(self, model, loader, true_labels):
        """
        Performs final testing on the test set.
        :param model: The best trained PyTorch model.
        :param loader: The test DataLoader.
        :param true_labels: Ground truth labels for the test set.
        :return: (predictions, true_labels, results_dict)
        """
        model.eval()
        predictions = []

        with torch.no_grad():
            count = 0
            for batch in loader:
                input_ids = batch["gene_ids"].to(self.device)
                input_vals = batch["values"].to(self.device)
                pad_mask = input_ids.eq(self.vocab[self.pad_token])

                sorted_layer_idx = batch.get("sorted_layer_idx")
                # Move sorted layer index to device if needed
                if self.args.graph_sort and self.args.layer_emb:
                    sorted_layer_idx = sorted_layer_idx.to(self.device)
                else:
                    sorted_layer_idx = None

                with torch.cuda.amp.autocast(enabled=False):
                    output = model(
                        input_ids,
                        input_vals,
                        src_key_padding_mask=pad_mask,
                        CLS=self.args.CLS,
                        CCE=self.args.CCE,
                        sorted_layer_idx=sorted_layer_idx
                    )

                logits = output["cls_output"]
                preds = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()

                predictions.append(preds)
                count += len(preds)

        predictions = np.concatenate(predictions, axis=0)

        # Calculate final metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average="macro")
        recall = recall_score(true_labels, predictions, average="macro")
        f1 = f1_score(true_labels, predictions, average="macro")

        self.logger.info(
            f"[Test] Acc: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"
        )

        results = {
            "test/accuracy": accuracy,
            "test/precision": precision,
            "test/recall": recall,
            "test/macro_f1": f1
        }

        return predictions, true_labels, results

    def _organizeResults(self, predictions, labels, results, data_configs):
        """
        Organizes final results, adds predictions to the raw anndata object, and saves results to a pickle file.
        :param predictions: Array of predicted labels (indices).
        :param labels: Array of true labels (indices).
        :param results: Dictionary of final metrics.
        :param data_configs: Dictionary containing anndata object and label maps.
        """
        adata = data_configs["adata_test_raw"]
        id2type = data_configs["id2type"]

        # Map prediction indices back to cell type names and store in anndata.obs
        adata.obs["predictions"] = [id2type[p] for p in predictions]

        # Save all relevant results and maps to a pickle file
        with open(self.save_dir / "results.pkl", "wb") as f:
            pickle.dump({
                "predictions": predictions,
                "labels": labels,
                "results": results,
                "id_maps": id2type
            }, f)

    def runAnnotation(self):
        """
        Main method to run the entire annotation process: load, train, validate, test, and save.
        """
        model, train_loader, valid_loader, test_loader, data_configs = self.load_data_and_model()
        self._loadCriterionAndOpt(model)

        best_f1 = 0
        best_model = None
        best_epoch = -1

        for epoch in range(1, self.args.epochs + 1):
            self._train(model, train_loader, epoch)

            # Validate the model
            mse, err, acc, prec, rec, f1 = self._evaluate(model, valid_loader, epoch)

            # Track the best model based on macro F1 score
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                best_model = copy.deepcopy(model)  # Deep copy to save model state
                self.logger.info(
                    f"[Best] Epoch {epoch} | New best f1: {best_f1:.4f} "
                    f"(f1={f1:.4f}, acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f})"
                )
            # Scheduler step to adjust learning rate
            self.scheduler.step()


        self.logger.info(f"[Summary] Best f1 {best_f1:.4f} achieved at epoch {best_epoch}")

        # Test the best model and save results
        preds, labels, results = self._test(best_model, test_loader, data_configs["test_labels"])
        self._organizeResults(preds, labels, results, data_configs)

        torch.save(best_model.state_dict(), self.save_dir / "best_model.pt") # Save final best model weights
        gc.collect() # Trigger garbage collection

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()
    task = AnnoTaskMamba(args.config_file)
    task.runAnnotation()

def check_model_weights(model):
    """
    Utility function to check for NaN or Inf values in model parameters.
    """
    has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"[NaN] detected in parameter: {name}")
            has_nan = True
        if torch.isinf(param).any():
            print(f"[Inf] detected in parameter: {name}")
            has_nan = True
    if not has_nan:
        print("✅ All model weights are finite (no NaN/Inf found).")
    return has_nan