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
            model_configs["nheads"] = args.nhead  # number of heads in nn.MultiheadAttention
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
        model_configs["nheads"] = args.nhead  # number of heads in nn.MultiheadAttention
        #dropout = args.dropout  # dropout probability
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
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        ckpt_emb_shape = pretrained_dict['encoder.embedding.weight'].size()
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        if logger is not None and not 'encoder.embedding.weight' in pretrained_dict:
            logger.warning(f'{"!" * 30}Embeddings Unavailable{"!" * 30}\n'
                           f'Expected shape: {model_dict["encoder.embedding.weight"].size()}\n'
                           f'But got shape: {ckpt_emb_shape} from ckpt {model_file}')
        # if logger is not None:
        #     for k, v in pretrained_dict.items():
        #         logger.info(f"Loading params {k} with shape {v.shape}")
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