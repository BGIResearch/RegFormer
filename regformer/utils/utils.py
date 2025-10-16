# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:utils.py
# @Software:PyCharm
# @Created Time:2024/2/26 5:32 PM
import torch,random,os
import logging
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from anndata import AnnData
import scib
from .. import logger
import munch
import wandb
from regformer.data.gene_tokenizer import GeneVocab
import toml
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def train_test_split_adata(adata, test_size=0.2):
    cell_indices = adata.obs.index
    cell_indices = cell_indices[~cell_indices.duplicated(keep='first')]
    train_indices, test_indices = train_test_split(cell_indices, test_size=test_size, random_state=42)
    train_data = adata[train_indices, :].copy()
    test_data = adata[test_indices, :].copy()
    return train_data, test_data


def get_reduced(tensor, current_device, dest_device, world_size):
    """
    将不同GPU上的变量或tensor集中在主GPU上，并得到均值
    """
    tensor = tensor.clone().detach() if torch.is_tensor(tensor) else torch.tensor(tensor)
    tensor = tensor.to(current_device)
    torch.distributed.reduce(tensor, dst=dest_device)
    tensor_mean = tensor.item() / world_size
    return tensor_mean

def seed_all(seed_value, cuda_deterministic=False):
    """
    设置所有的随机种子
    """
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
def add_file_handler(logger: logging.Logger, log_file_path: Path):
    """
    Add a file handler to the logger.
    """
    h = logging.FileHandler(log_file_path)

    # format showing time, name, function, and message
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(funcName)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    h.setFormatter(formatter)
    h.setLevel(logger.level)
    logger.addHandler(h)

def eval_scib_metrics(
    adata: AnnData,
    batch_key: str = "str_batch",
    label_key: str = "celltype",
    embed_key: str = "X_scGPT",
    notes: Optional[str] = None,
) -> Dict:
    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key=batch_key,
        label_key=label_key,
        embed=embed_key,
        isolated_labels_asw_=False,
        silhouette_=True,
        hvg_score_=False,
        graph_conn_=True,#T
        pcr_=True,#T
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=True,#T  # use the clustering, bias to the best matching
        ari_=True,#T # use the clustering, bias to the best matching
        cell_cycle_=False,
        kBET_=False,  # kBET return nan sometimes, need to examine
        ilisi_=False,
        clisi_=False,
    )

    if notes is not None:
        logger.info(f"{notes}")

    logger.info(f"{results}")

    result_dict = results[0].to_dict()
    logger.info(
        "Biological Conservation Metrics: \n"
        f"ASW (cell-type): {result_dict['ASW_label']:.4f}, graph cLISI: {result_dict['cLISI']:.4f}, "
        f"isolated label silhouette: {result_dict['isolated_label_silhouette']:.4f}, \n"
        "Batch Effect Removal Metrics: \n"
        f"PCR_batch: {result_dict['PCR_batch']:.4f}, ASW (batch): {result_dict['ASW_label/batch']:.4f}, "
        f"graph connectivity: {result_dict['graph_conn']:.4f}, graph iLISI: {result_dict['iLISI']:.4f}"
    )

    result_dict["avg_bio"] = np.mean(
        [
            result_dict["NMI_cluster/label"],
            result_dict["ARI_cluster/label"],
            result_dict["ASW_label"],
        ]
    )

    # remove nan value in result_dict
    result_dict = {k: v for k, v in result_dict.items() if not np.isnan(v)}

    return result_dict

def load_config(config_file):
    args = munch.munchify(toml.load(config_file))
    # if args.model_name in ('gpt', 'mamba'):
    #     with open(args.model_param_file, 'r') as fd:
    #         params = json.load(fd)
    #     for p in params:
    #         if p not in args:
    #             args[p] = params[p]
    return args


def model_config(args,is_master=True):
    if args.load_model:
        model_dir = Path(args.load_model)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"

        # model
        if os.path.exists(model_config_file):
            with open(model_config_file, "r") as f:
                model_configs = json.load(f)
        else:
            model_configs = {}
            model_configs["embsize"] = args.layer_size  # embedding dimension
            model_configs["d_hid"] = args.layer_size  # dimension of the feedforward network in TransformerEncoder
            model_configs["nlayers"] = args.nlayers  # number of TransformerEncoderLayer in TransformerEncoder
            # dropout = args.dropout  # dropout probability
        if is_master:
            logger.info(
                f"Resume model from {model_file}, the model args will override the "
                f"config {model_config_file}."
            )
        # embsize = model_configs["embsize"]
        # nhead = model_configs["nheads"]
        # d_hid = model_configs["d_hid"]
        # nlayers = model_configs["nlayers"]
        # n_layers_cls = model_configs["n_layers_cls"]
    else:
        model_configs={}
        model_configs["embsize"] = args.layer_size  # embedding dimension
        model_configs["d_hid"] = args.layer_size  # dimension of the feedforward network in TransformerEncoder
        model_configs["nlayers"] = args.nlayers  # number of TransformerEncoderLayer in TransformerEncoder
        model_configs["n_layers_cls"] = 3
        vocab_file = args.vocab_file
        model_file = None
    # vocab = GeneVocab.from_file(vocab_file)
    return model_configs,vocab_file,model_file

def load_ckpt(model,model_file,args,logger=None):
    try:
        model.load_state_dict(torch.load(model_file))
        if logger is not None:
            logger.info(f"Loading all model params from {model_file}")
    except:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)

        missed_keys = []
        for k, v in pretrained_dict.items():
            if k not in model_dict:
                missed_keys.append(f"[Missing] {k} not in model")
            elif v.shape != model_dict[k].shape:
                missed_keys.append(f"[Shape Mismatch] {k}: {v.shape} vs {model_dict[k].shape}")

        if missed_keys:
            print("⚠️ The following keys were not loaded from pretrained model:")
            for msg in missed_keys:
                print(msg)
        else:
            print("✅ All keys loaded successfully.")

        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    pre_freeze_param_count = sum(
        dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
    # Freeze all pre-decoder weights
    for name, para in model.named_parameters():
        if args.freeze and "encoder" in name and "transformer_encoder" not in name:
            # if args.freeze and "encoder" in name:
            print(f"freezing weights for: {name}")
            para.requires_grad = False
    post_freeze_param_count = sum(
        dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
    if logger is not None:
        logger.info(f"Total Pre freeze Params {(pre_freeze_param_count)}")
        logger.info(f"Total Post freeze Params {(post_freeze_param_count)}")
    return model

def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


def print_model_size(model):
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    trainable_dict = {}
    non_trainable_dict = {}
    for name, param in model.named_parameters():
        tem = np.prod(param.size())
        total_params += tem
        if param.requires_grad:
            trainable_params += tem
            trainable_dict[name] = tem
        else:
            non_trainable_params += tem
            non_trainable_dict[name] = tem
    print(f'Total params: {total_params / 1e6}M')
    print(f'Trainable params: {trainable_params / 1e6}M')
    print(f'Non-trainable params: {non_trainable_params / 1e6}M')
    # print(f"*******Trainable info********")
    # for name in trainable_dict:
    #     print(f"Parameter name: {name}, Size: {trainable_dict[name]}")
    # print(f"*******Non-Trainable info********")
    for name in non_trainable_dict:
        print(f"Parameter name: {name}, Size: {non_trainable_dict[name]}")

def gene2vec_embedding(g2v_file, g2v_genes):
    gene2vec_weight = np.load(g2v_file)
    gene_emb_dict = {}
    with open(g2v_genes, 'r') as fd:
        gene_list = [line.strip('\n') for line in fd]
    for i in range(len(gene_list)):
        gene_emb_dict[gene_list[i]] = gene2vec_weight[i]
    return gene_emb_dict
def refine_embedding(X, batch_labels, class_labels, type_weight=1.0):
    """
    Remove batch shift by aligning each (batch, celltype) group to its global
    celltype mean, and optionally amplify the distance between different celltype means.

    Parameters:
    ----------
    X : np.ndarray, shape (n_cells, n_features)
        Embedding matrix (e.g., from FC layer).
    batch_labels : array-like, shape (n_cells,)
        Batch assignment per cell.
    class_labels : array-like, shape (n_cells,)
        Cell type label per cell.
    type_weight : float, default=1.0
        Factor to scale inter-celltype distance. Use >1.0 to amplify, <1.0 to compress.

    Returns:
    -------
    X_corrected : np.ndarray, shape (n_cells, n_features)
        Batch-corrected and class-enhanced embedding.
    """
    X = np.asarray(X)
    batch_labels = np.asarray(batch_labels)
    class_labels = np.asarray(class_labels)
    X_corrected = X.copy()

    unique_celltypes = np.unique(class_labels)
    unique_batches = np.unique(batch_labels)

    # 1. Cell type global means
    type_means = {}
    for c in unique_celltypes:
        idx = np.where(class_labels == c)[0]
        if len(idx) < 2:
            continue
        type_means[c] = np.mean(X[idx], axis=0)

    # 2. Batch-wise alignment (same as before)
    for c in unique_celltypes:
        if c not in type_means:
            continue
        global_mean = type_means[c]
        for b in unique_batches:
            idx = np.where((batch_labels == b) & (class_labels == c))[0]
            if len(idx) < 2:
                continue
            batch_mean = np.mean(X[idx], axis=0)
            shift = batch_mean - global_mean
            X_corrected[idx] -= shift

    # 3. Amplify inter-class differences
    grand_mean = np.mean(X_corrected, axis=0)
    for c in unique_celltypes:
        idx = np.where(class_labels == c)[0]
        if len(idx) < 2:
            continue
        # Push class mean away from grand mean
        class_mean = np.mean(X_corrected[idx], axis=0)
        offset = (class_mean - grand_mean) * (type_weight - 1.0)
        X_corrected[idx] += offset

    return X_corrected
def aggregate_gene_embeddings(loader, model, layer_size, vocab, pad_token, device):
    gene_sum = torch.zeros(len(vocab), layer_size, device=device)
    gene_count = torch.zeros(len(vocab), device=device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        for batch in tqdm(loader, desc="Aggregating gene embeddings"):
            input_ids = batch["gene_ids"].to(device)
            input_vals = batch["values"].to(device)
            pad_mask = input_ids.eq(vocab[pad_token])

            sorted_layer_idx = batch.get("sorted_layer_idx")
            if sorted_layer_idx is not None:
                sorted_layer_idx = sorted_layer_idx.to(device)

            out = model._encode(
                src=input_ids,
                values=input_vals,
                batch_labels=None,
                src_key_padding_mask=pad_mask,
                sorted_layer_idx=sorted_layer_idx
            )  # [B, L, Z]

            mask = ~pad_mask
            for b in range(out.shape[0]):
                ids = input_ids[b][mask[b]]
                emb = out[b][mask[b]]
                gene_sum[ids] += emb
                gene_count[ids] += 1

    gene_embeddings = gene_sum / (gene_count.unsqueeze(-1) + 1e-9)
    valid_gene_ids = torch.nonzero(gene_count > 0, as_tuple=True)[0].cpu()

    return gene_embeddings.cpu(), valid_gene_ids