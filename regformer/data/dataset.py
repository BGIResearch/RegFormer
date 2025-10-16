# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:dataset.py
# @Software:PyCharm
# @Created Time:2023/12/28 5:56 PM
import os, sys
from typing import Dict, Tuple, Union
import scanpy as sc
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import lmdb
import dgl
import torch
import os.path as osp
import json
import anndata
from scanpy.get import _get_obs_rep
from scipy.sparse import issparse
from tqdm import tqdm
import random
import copy
from regformer.data.preprocess import Preprocessor
from regformer.data.gene_tokenizer import tokenize_and_pad_batch_v2


def Load_Data(data_path, args, **kwargs):
    if args.task == 'Cell_annotation':
        return cell_annotation_dataset(data_path=data_path, args=args, **kwargs)
    elif args.task == 'Integration':
        return Integration_dataset(data_path=data_path, args=args, **kwargs)
    elif args.task == 'Pretraining':
        return Pretraining_dataset(data_path=data_path, args=args, **kwargs)
    elif args.task == 'GRN_inference':
        raise NotImplementedError


def cell_annotation_dataset(data_path, args, **kwargs):
    '''
    Args:
        data_path:
        args:
        **kwargs:
    Returns:
    '''
    data_is_raw = args.data_is_raw
    filter_gene_by_counts = args.filter_gene_by_counts
    if data_path.endswith('h5ad'):
        adata = sc.read(data_path)
    else:
        data_dir = Path(os.path.join(args.data_path, args.task, args.data_name))
        adata = sc.read(data_dir / f"{args.data_name}.h5ad")
    adata.obs["celltype"] = adata.obs[args.cell_type_column].astype("category")
    if args.gene_column != 'none':
        adata.var = adata.var.set_index(args.gene_column)
    adata.var["gene_name"] = adata.var.index.tolist()
    if args.umap_column != 'none' and 'X_umap' not in adata.obsm:
        if args.umap_column == 'TSNE':
            adata.obsm["X_umap"] = np.array(adata.obs.loc[:, ["TSNE.1", "TSNE.2"]])
        else:
            adata.obsm["X_umap"] = adata.obsm[args.umap_column]
    if args.pca_column != 'none' and 'X_pca' not in adata.obsm:
        adata.obsm["X_pca"] = adata.obsm[args.pca_column]

    celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
    celltypes = adata.obs["celltype"].unique()
    num_types = len(np.unique(celltype_id_labels))
    id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
    adata.obs["celltype_id"] = celltype_id_labels
    train_idx, valid_idx = train_test_split(range(adata.n_obs), test_size=args.test_size, random_state=args.seed)

    adata_test = adata[valid_idx].copy()
    adata_test_raw = adata_test.copy()
    adata = adata[train_idx].copy()

    adata.obs["batch_id"] = 0
    adata_test.obs["batch_id"] = 1

    logger = kwargs['logger']
    vocab = kwargs['vocab']
    is_master = kwargs['is_master']
    mask_value = kwargs['mask_value']
    pad_value = kwargs['pad_value']
    pad_token = kwargs['pad_token']

    ## only retain the gene that appears in vocab
    adata, _ = filter_gene(vocab=vocab, adata=adata, is_master=is_master, logger=logger)
    adata_test, _ = filter_gene(vocab=vocab, adata=adata_test, is_master=is_master, logger=logger)

    # set up the preprocessor, use the args to analysis_tools the workflow
    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=filter_gene_by_counts,  # step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=args.data_is_raw,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=args.data_is_raw,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg = args.n_hvg if args.n_hvg > 0 and args.n_hvg < adata.n_vars else False,  # 5. whether to subset the raw data to highly variable genes
        use_cell_type=args.use_cell_type,
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=args.n_bins,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )

    adata = preprocessor(adata, batch_key=None).copy()

    adata_test = adata_test[:, adata.var_names].copy()

    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=False,  # step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=args.data_is_raw,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=args.data_is_raw,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3",
        binning=args.n_bins,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )
    adata_test = preprocessor(adata_test, batch_key=None).copy()

    print(adata_test)
    print(adata)

    if args.max_seq_len < 0:
        q_len_train = compute_nonzero_len_quantile(adata, q=0.9)
        q_len_test = compute_nonzero_len_quantile(adata_test, q=0.9)

        args.max_seq_len = int(max(q_len_train, q_len_test))
        print(f"📏 max_seq_len set to 90% quantile: {args.max_seq_len}")

    train_data_pt, valid_data_pt, test_data_pt, num_batch_types = prepare_cell_data(adata=adata, adata_test=adata_test,
                                                                                    args=args,
                                                                                    vocab=vocab, is_master=is_master,
                                                                                    mask_value=mask_value,
                                                                                    pad_value=pad_value, logger=logger,
                                                                                    sort_seq_batch=False,
                                                                                    pad_token=pad_token)

    return train_data_pt, valid_data_pt, test_data_pt, num_batch_types, celltypes, id2type, num_types, adata_test_raw


def infer_dataset(data_path=None, adata=None, args=None, **kwargs):
    '''
    Args:
        data_path:
        args:
        **kwargs:
    Returns:
    '''
    data_is_raw = args.data_is_raw
    filter_gene_by_counts = args.filter_gene_by_counts
    if adata is None:
        if data_path.endswith('h5ad'):
            adata = sc.read(data_path)
        else:
            data_dir = Path(os.path.join(args.data_path, args.task, args.data_name))
            adata = sc.read(data_dir / f"{args.data_name}.h5ad")
    adata.obs["celltype"] = adata.obs[args.cell_type_column].astype("category")
    if args.gene_column != 'none':
        adata.var = adata.var.set_index(args.gene_column)
    adata.var["gene_name"] = adata.var_names.tolist()

    celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
    adata.obs["celltype_id"] = celltype_id_labels
    adata.obs["batch_id"] = adata.obs[args.batch_column].astype("category").cat.codes if args.batch_column else 0
    logger = kwargs['logger']
    vocab = kwargs['vocab']
    is_master = kwargs['is_master']
    mask_value = kwargs['mask_value']
    pad_value = kwargs['pad_value']
    pad_token = kwargs['pad_token']

    ## only retain the gene that appears in vocab
    adata, _ = filter_gene(vocab=vocab, adata=adata, is_master=is_master,
                           logger=logger)  # only retain the gene that appears in vocab
    # set up the preprocessor, use the args to analysis_tools the workflow
    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=filter_gene_by_counts,  # step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=args.data_is_raw,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=args.data_is_raw,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=args.n_hvg if args.n_hvg > 0 and args.n_hvg < adata.n_vars else False,  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=args.n_bins,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )

    adata = preprocessor(adata, batch_key=None).copy()

    if args.max_seq_len < 0:
        q_len = compute_nonzero_len_quantile(adata, q=0.9)
        args.max_seq_len = int(q_len)
        print(f"📏 max_seq_len set to 90% quantile: {args.max_seq_len}")

    train_data_pt = prepare_cell_data_for_infer(adata=adata,
                                      args=args,
                                      vocab=vocab, is_master=is_master,
                                      mask_value=mask_value,
                                      pad_value=pad_value, logger=logger,
                                      sort_seq_batch=False,
                                      pad_token=pad_token)

    return train_data_pt


def Integration_dataset(data_path, args, **kwargs):
    if data_path.endswith('h5ad'):
        adata = sc.read(data_path)
    else:
        data_dir = Path(os.path.join(args.data_path, args.task, args.data_name))
        adata = sc.read(data_dir / f"{args.data_name}.h5ad")

    adata.obs["celltype"] = adata.obs[args.cell_type_column].astype("category")
    celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
    adata.obs["celltype_id"] = celltype_id_labels
    num_cell_types = len(np.unique(celltype_id_labels))
    id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))

    # %%
    # make the batch category column, used in test data
    adata.obs["str_batch"] = adata.obs[args.batch_column].astype(str)
    batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels
    if args.gene_column != 'none':
        adata.var = adata.var.set_index(args.gene_column)
    adata.var["gene_name"] = adata.var.index.tolist()

    logger = kwargs['logger']
    vocab = kwargs['vocab']
    is_master = kwargs['is_master']
    mask_value = kwargs['mask_value']
    pad_value = kwargs['pad_value']
    pad_token = kwargs['pad_token']

    ## only retain the gene that appears in vocab
    adata, _ = filter_gene(vocab=vocab, adata=adata, is_master=is_master,
                           logger=logger)  # only retain the gene that appears in vocab

    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=3,  # step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=args.data_is_raw,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=getattr(args, "n_hvg", False),  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3" if args.data_is_raw else "cell_ranger",
        binning=args.n_bins,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
        verbose=True
    )
    preprocessor(adata, batch_key="str_batch" if args.data_name != "heart_cell" else None)
    if args.per_seq_batch_sample:
        # sort the adata by batch_id in advance
        adata_test = adata[adata.obs["batch_id"].argsort()].copy()
    else:
        adata_test = adata.copy()
    train_data_pt, valid_data_pt, test_data_pt, num_batch_types = prepare_cell_data(adata=adata, adata_test=adata_test,
                                                                                    args=args, vocab=vocab,
                                                                                    is_master=is_master,
                                                                                    mask_value=mask_value,
                                                                                    pad_value=pad_value, logger=logger,
                                                                                    sort_seq_batch=False,
                                                                                    pad_token=pad_token)
    return train_data_pt, valid_data_pt, test_data_pt, num_batch_types, adata_test, num_cell_types, id2type

def is_lmdb_dir(path):
    """
    Check whether the given path is a valid LMDB directory
    (i.e., a directory that contains a data.mdb file).
    """
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, 'data.mdb'))

def get_lmdb_dataset(path_or_dir, args, **kwargs):
    """
    Automatically determine whether to use LMDBDataset or LMDBListDataset
    based on the path structure:
    - If it's a single LMDB directory with data.mdb → use LMDBDataset.
    - If it's a directory containing multiple LMDB subfolders → use LMDBListDataset.
    """
    if is_lmdb_dir(path_or_dir):
        return LMDBDataset(db_path=path_or_dir, bin_num=args.n_bins, args=args, **kwargs)
    elif os.path.isdir(path_or_dir):
        # Collect all valid LMDB subdirectories that contain data.mdb
        lmdb_dirs = [
            os.path.join(path_or_dir, d)
            for d in os.listdir(path_or_dir)
            if is_lmdb_dir(os.path.join(path_or_dir, d))
        ]
        if not lmdb_dirs:
            raise ValueError(f"No LMDB subfolders (with data.mdb) found in {path_or_dir}")
        return LMDBListDataset(db_paths=lmdb_dirs, args=args, **kwargs)
    else:
        raise ValueError(f"Invalid LMDB path: {path_or_dir}")

def Pretraining_dataset(args, **kwargs):
    if args.data_name == 'cellxgene':
        train_data = get_lmdb_dataset(args.lmdb_path, args, **kwargs)

        valid_args = copy.deepcopy(args)
        valid_args.data_name = 'cellxgene'
        valid_data = get_lmdb_dataset(args.val_lmdb_path, valid_args, **kwargs)

        return train_data, valid_data

    else:
        raise ValueError(f'Invalid {args.task} dataset {args.data_name}')


from torch.utils.data import Dataset

class LMDBListDataset(Dataset):
    def __init__(self, db_paths, args, **kwargs):
        """
        db_paths: List of LMDB database paths (e.g. ['path/to/train1.db', 'path/to/train2.db'])
        args: Shared arguments for all LMDB datasets
        kwargs: Shared kwargs (e.g., vocab, mask_token, etc.)
        """
        self.datasets = []
        self.cumulative_lengths = []

        cum_len = 0
        for path in db_paths:
            dataset = LMDBDataset(db_path=path, args=args, **kwargs)
            self.datasets.append(dataset)
            cum_len += len(dataset)
            self.cumulative_lengths.append(cum_len)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, index):
        # Find the dataset that contains this index
        for i, cum_len in enumerate(self.cumulative_lengths):
            if index < cum_len:
                if i == 0:
                    local_index = index
                else:
                    local_index = index - self.cumulative_lengths[i - 1]
                return self.datasets[i][local_index]

        raise IndexError("Index out of bounds for LMDBListDataset")


class LMDBDataset(Dataset):
    def __init__(self, db_path, args=None, pad_value=-2, **kwargs):
        super().__init__()
        self.vocab = kwargs['vocab']
        self.args = args
        self.invalid_datapoint_count = 0
        self.pad_value = pad_value
        self.mask_token = kwargs['mask_token']
        self.unk_token = kwargs['unk_token']
        if args.data_name == 'panglao':
            self.gene_idx_array = np.array(np.load(args.gene_array_file, allow_pickle=True))
        else:
            self.gene_idx_array = None
        self.n_bins = args.n_bins
        self.mask_ratio = kwargs['mask_ratio']
        self.append_cls = kwargs['append_cls']
        self.include_zero_gene = kwargs["include_zero_gene"]
        self.max_seq_len = kwargs["max_seq_len"]
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)
        self.graph_sort = args.graph_sort
        self.sampling_etype = args.sampling_etype
        self.sampling_g = None
        if self.graph_sort:
            self.layer_mask = args.layer_mask
            self.grn = dgl.load_graphs(args.graph_path)[0][0]
            if args.data_name == 'panglao':
                self.valid_idx = self.gene_idx_array < self.grn.num_nodes()  # TODO:delete this if new dataset is valid
                self.gene_idx_array = self.gene_idx_array[self.valid_idx]  # TODO:delete this if new dataset is valid

    def __getitem__(self, index):
        values = self.txn.get(u'{}'.format(index).encode())
        try:
            if self.args.data_name == 'panglao':
                values = np.frombuffer(values)  # np.array([gene_num,])
                values = values[self.valid_idx]  # TODO:delete this if new dataset is valid
                gene_ids = self.gene_idx_array
            else:
                datapoint = json.loads(values)  # ['express_x','organ','celltype','sequence','disease','gene_index']
                values = np.array(datapoint['express_x'])
                gene_ids = np.frombuffer(self.txn.get(key=u'{}'.format(datapoint['gene_index']).encode()),
                                         dtype=np.int64)
                if len(values) != len(gene_ids):
                    gene_ids = np.random.choice(range(0, len(self.vocab) - 10), size=(len(values),),
                                                replace=False).astype(np.int64)
                    self.invalid_datapoint_count += 1
        except:
            self.invalid_datapoint_count += 1
            gene_ids = np.random.choice(range(0, len(self.vocab) - 10), size=(self.max_seq_len * 2,),
                                        replace=False).astype(np.int64)
            values_nz = np.random.uniform(0, 5, size=(int(self.max_seq_len * 0.1),)).astype(np.float64)
            values = np.zeros_like(gene_ids, dtype=np.float64)
            values[:int(self.max_seq_len * 0.1)] = values_nz
            np.random.shuffle(values)
        assert len(values) == len(gene_ids)
        if self.n_bins > 0:
            input_values, bin_edge = self._binning(values)
        else:
            input_values = values
        values, gene_ids, masked_values, sorted_gene_ids, masked_sorted_gene_ids, sorted_layer_idx = self._pad_and_mask(
            input_values, gene_ids=gene_ids)
        datapoint = {"gene_ids": gene_ids, "masked_values": masked_values, "target_values": values,
                     "sorted_gene_ids": sorted_gene_ids, "masked_sorted_gene_ids": masked_sorted_gene_ids,
                     "sorted_layer_idx": sorted_layer_idx}
        return datapoint

    def __len__(self):
        return int(self.txn.get(b'__len__').decode("utf-8"))
        # return 1000

    def _binning(self, values):
        non_zero_ids = values.nonzero()
        non_zero_row = values[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, self.n_bins - 1))
        # bins = np.sort(np.unique(bins))
        # NOTE: comment this line for now, since this will make the each category
        # has different relative meaning across datasets
        non_zero_digits = self._digitize(non_zero_row, bins)
        assert non_zero_digits.min() >= 1
        assert non_zero_digits.max() <= self.n_bins - 1
        binned_row = np.zeros_like(values, dtype=np.int64).copy()
        binned_row[non_zero_ids] = non_zero_digits
        bin_edge = np.concatenate([[0], bins])
        return binned_row, bin_edge

    def _digitize(self, x: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """
        Digitize the data into bins. This method spreads data uniformly when bins
        have same values.

        Args:

        x (:class:`np.ndarray`):
            The data to digitize.
        bins (:class:`np.ndarray`):
            The bins to use for digitization, in increasing order.

        Returns:

        :class:`np.ndarray`:
            The digitized data.
        """
        assert x.ndim == 1 and bins.ndim == 1

        left_digits = np.digitize(x, bins)
        right_difits = np.digitize(x, bins, right=True)

        rands = np.random.rand(len(x))  # uniform random numbers

        digits = rands * (right_difits - left_digits) + left_digits
        digits = np.ceil(digits).astype(np.int64)
        return digits

    def _pad_and_mask(self, values, gene_ids):
        if self.include_zero_gene:
            values = values
            gene_ids = gene_ids
        else:
            idx = np.nonzero(values)[-1]
            values = np.array(values[idx])
            gene_ids = np.array(gene_ids[idx])

        if len(gene_ids) > self.max_seq_len:
            idx = np.random.choice(len(gene_ids), self.max_seq_len, replace=False)
            gene_ids = gene_ids[idx]
            values = values[idx]
            # masked_values=masked_values[idx]
        masked_values = random_mask_value(values, self.mask_ratio)

        if self.graph_sort:
            if 'random_sort' in self.args and self.args.random_sort:
                sorted_gene_ids, sorted_layer_idx, _ = self.random_sorting(gene_ids,
                                                                           values=None)  # the length here would <=max_len
                gene_ids, layer_idx, values = self.random_sorting(gene_ids, values)
            else:
                sorted_gene_ids, sorted_layer_idx, _ = self.topological_sorting(gene_ids, values=None,
                                                                                sample=False)  # the length here would <=max_len
                gene_ids, layer_idx, values = self.topological_sorting(gene_ids, values, sample=False)
            mask_id = self.vocab[self.mask_token]
            if sorted_gene_ids.__len__() < self.max_seq_len:
                pad_id = self.vocab['<pad>']
                pad_layer_idx = 0
                sorted_gene_ids = np.concatenate([sorted_gene_ids,
                                                  np.full(self.max_seq_len - len(sorted_gene_ids), pad_id,
                                                          dtype=sorted_gene_ids.dtype)])
                sorted_layer_idx = np.concatenate([sorted_layer_idx,
                                                   np.full(self.max_seq_len - len(sorted_layer_idx), pad_layer_idx,
                                                           dtype=sorted_layer_idx.dtype)])
            if self.layer_mask:
                selected_masked_layer = random.sample(range(1, max(sorted_layer_idx) + 1),
                                                      min(1, int(max(sorted_layer_idx) * self.mask_ratio)))
                assert selected_masked_layer.__len__() < max(sorted_layer_idx)
                masking_position = np.isin(sorted_layer_idx, selected_masked_layer)
                masked_sorted_gene_ids = sorted_gene_ids.copy()
                masked_sorted_gene_ids[masking_position] = mask_id
            else:
                masked_sorted_gene_ids = random_mask_value(values=sorted_gene_ids, mask_ratio=self.mask_ratio,
                                                           mask_value=self.vocab[self.mask_token],
                                                           pad_value=self.vocab['<pad>'])
                masked_sorted_gene_ids = torch.from_numpy(masked_sorted_gene_ids)

        ## padding
        if len(gene_ids) < self.max_seq_len:
            pad_id = self.vocab['<pad>']
            gene_ids = np.concatenate(
                [gene_ids, np.full(self.max_seq_len - len(gene_ids), pad_id, dtype=gene_ids.dtype)])
            values = np.concatenate(
                [values, np.full(self.max_seq_len - len(values), self.pad_value, dtype=values.dtype)])
            masked_values = np.concatenate(
                [masked_values,
                 np.full(self.max_seq_len - len(masked_values), self.pad_value, dtype=masked_values.dtype)])

        if self.append_cls:
            values = np.insert(values, self.max_seq_len, 0)
            gene_ids = np.insert(gene_ids, self.max_seq_len, self.vocab['<cls>'])
            masked_values = np.insert(masked_values, self.max_seq_len, 0)
            if self.graph_sort:
                masked_sorted_gene_ids = np.insert(masked_sorted_gene_ids, self.max_seq_len, self.vocab['<cls>'])
                sorted_gene_ids = np.insert(sorted_gene_ids, self.max_seq_len, self.vocab['<cls>'])
                sorted_layer_idx = np.insert(sorted_layer_idx, self.max_seq_len, 0)

        if self.graph_sort:
            masked_sorted_gene_ids = torch.tensor(masked_sorted_gene_ids).int()
            sorted_gene_ids = torch.tensor(sorted_gene_ids).int()
            sorted_layer_idx = torch.tensor(sorted_layer_idx).int()
        else:
            masked_sorted_gene_ids = 0
            sorted_gene_ids = 0
            sorted_layer_idx = 0

        return torch.tensor(values).float(), torch.tensor(gene_ids).int(), torch.tensor(
            masked_values).float(), sorted_gene_ids, masked_sorted_gene_ids, sorted_layer_idx

    def topological_sorting(self, gene_ids, values, sample=False):
        if sample and (len(gene_ids) < self.max_seq_len):
            assert self.sampling_g is not None
            sub_g = dgl.sampling.sample_neighbors(self.sampling_g, nodes={'gene': torch.tensor(gene_ids)}, fanout=5,
                                                  edge_dir='out')
            unique_node = torch.cat([torch.tensor(gene_ids), sub_g.edges(order='srcdst')[0],
                                     sub_g.edges(order='srcdst')[1]]).unique().tolist()
            # remove the isolate&not_ori node
            sub_grn = dgl.node_subgraph(self.grn, unique_node)
            is_isolate = np.array(torch.logical_and(sub_grn.in_degrees() == 0, sub_grn.out_degrees() == 0))
            is_ori = np.isin(np.array(sub_grn.ndata[dgl.NID]), gene_ids)
            valid_node = sub_grn.ndata['_ID'][torch.from_numpy(~np.logical_and(is_isolate, ~is_ori))]
            if len(valid_node) > self.max_seq_len:
                valid_graph = dgl.node_subgraph(self.grn, gene_ids)
            else:
                valid_graph = dgl.node_subgraph(self.grn, valid_node)

        else:
            valid_graph = dgl.node_subgraph(self.grn, gene_ids)

        topo_sorting = dgl.topological_nodes_generator(valid_graph)
        sort_layer_idx = []
        for idx, layer in enumerate(topo_sorting):
            sort_layer_idx += [idx + 1] * len(layer)
        sorted_index = torch.cat(topo_sorting)
        sorting_gene_ids = valid_graph.ndata['_ID'][sorted_index]
        if values is not None:
            sorting_values = np.array(values[sorted_index])
        else:
            sorting_values = None

        return np.array(sorting_gene_ids), np.array(sort_layer_idx), sorting_values

    def random_sorting(self, gene_ids, values):
        gene_ids = np.array(gene_ids)
        permuted_idx = np.random.permutation(len(gene_ids))
        sorting_gene_ids = gene_ids[permuted_idx]

        if values is not None:
            sorting_values = np.array(values)[permuted_idx]
        else:
            sorting_values = None
        max_layer = 10
        sort_layer_idx = np.random.randint(1, max_layer + 1, size=len(sorting_gene_ids))
        return sorting_gene_ids, sort_layer_idx, sorting_values


class H5adDataset(Dataset):
    def __init__(self, n_bins=51, source_dir='', result_binned_key="X_binned", prep_dir='', **kwargs):
        self.n_bins = n_bins
        h5ad_file_list = [file for file in os.listdir(source_dir) if file.endswith('.h5ad')]
        self.h5ad_file_list = h5ad_file_list
        self.prep_dir = prep_dir
        self.vocab = kwargs['vocab']
        self.mask_ratio = kwargs['mask_ratio']
        self.append_cls = kwargs['append_cls']
        self.include_zero_gene = kwargs["include_zero_gene"]
        self.max_seq_len = kwargs["max_seq_len"]
        print("Binning and filtering data ...")
        if not isinstance(n_bins, int):
            raise ValueError(
                "Binning arg must be an integer, but got {}.".format(n_bins)
            )
        self.length_list = []
        self.gene_num = []
        self.n_files = 0
        self.max_non_zero_count = 0
        self.min_non_zero_count = float('inf')
        for file in tqdm(self.h5ad_file_list):
            self.n_files += 1
            target_file = osp.join(prep_dir, file)
            if os.path.exists(target_file):
                if kwargs['need_length']:
                    adata = anndata.read_h5ad(target_file)
                    self.length_list.append(adata.n_obs)
                    self.gene_num.append(adata.n_vars)
                    self.max_non_zero_count = max((adata.X > 0).sum(axis=1).max(), self.max_non_zero_count)
                    self.min_non_zero_count = min((adata.X > 0).sum(axis=1).min(), self.min_non_zero_count)
                else:
                    self.length_list.append(0)
                    self.gene_num.append(0)
                continue
            ## filter genes that don't exist in vocab
            adata = anndata.read_h5ad(osp.join(source_dir, file))
            adata, _ = filter_gene(self.vocab, adata, False, None)
            ## binning
            binned_rows = []
            bin_edges = []
            layer_data = _get_obs_rep(adata, layer=None)  # Return values for observations in adata.
            layer_data = layer_data.A if issparse(layer_data) else layer_data
            for row in layer_data:
                non_zero_ids = row.nonzero()
                non_zero_row = row[non_zero_ids]
                bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
                # bins = np.sort(np.unique(bins))
                # NOTE: comment this line for now, since this will make the each category
                # has different relative meaning across datasets
                non_zero_digits = self._digitize(non_zero_row, bins)
                assert non_zero_digits.min() >= 1
                assert non_zero_digits.max() <= n_bins - 1
                binned_row = np.zeros_like(row, dtype=np.int64).copy()
                binned_row[non_zero_ids] = non_zero_digits
                binned_rows.append(binned_row)
                bin_edges.append(np.concatenate([[0], bins]))
            adata.layers[result_binned_key] = np.stack(binned_rows)
            adata.obsm["bin_edges"] = np.stack(bin_edges)
            self.length_list.append(adata.n_obs)
            self.gene_num.append(adata.n_vars)
            self.max_non_zero_count = max((adata.X > 0).sum(axis=1).max(), self.max_non_zero_count)
            self.min_non_zero_count = min((adata.X > 0).sum(axis=1).min(), self.min_non_zero_count)
            adata.write_h5ad(target_file)
        assert len(self.length_list) == len(self.h5ad_file_list)
        self.cumulative_sizes = np.cumsum(self.length_list)
        print("Binning completed!")

    def __len__(self):
        # Return the total number of samples across all files
        return np.sum(self.length_list)

    def __getitem__(self, idx):
        # Efficiently fetch a single item across the dataset
        if idx < 0 or idx >= self.__len__():
            raise IndexError
        file_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
        adjusted_idx = idx - (self.cumulative_sizes[file_idx - 1] if file_idx > 0 else 0)
        adata = anndata.read_h5ad(osp.join(self.prep_dir, self.h5ad_file_list[file_idx]))
        target_values, gene_ids, masked_values = self._tokenize_and_pad(adata[adjusted_idx], self.mask_ratio,
                                                                        self.max_seq_len)
        datapoint = {"gene_ids": gene_ids, "masked_values": masked_values, "target_values": target_values,
                     "cell_type": '<unk>'}
        return datapoint

    def _tokenize_and_pad(self, adata, mask_ratio, max_len):
        genes = adata.var_names.tolist()
        values = adata.layers['X_binned']
        if values.shape[1] != len(genes):
            raise ValueError(
                f"Number of features in data ({values.shape[1]}) does not match "
                f"number of gene_ids ({len(genes)})."
            )
        if self.include_zero_gene:
            values = values
            gene_ids = np.array(self.vocab(genes), dtype=int)
        else:
            idx = np.nonzero(adata.X)[-1]
            values = values[:, idx]
            gene_ids = np.array(self.vocab(genes), dtype=int)
            gene_ids = gene_ids[idx]
        if self.append_cls:
            values = np.insert(values, self.max_seq_len, 0)
            gene_ids = np.insert(gene_ids, self.max_seq_len, self.vocab['<cls>'])
        masked_value = torch.from_numpy(random_mask_value(values, mask_ratio)).float().view(1, -1)
        values, gene_ids = torch.tensor(values).view(1, -1), torch.tensor(gene_ids).view(1, -1)
        if len(gene_ids[-1]) > max_len:
            if not self.append_cls:
                idx = np.random.choice(len(gene_ids[-1]), max_len, replace=False)
            else:
                idx = np.random.choice(len(gene_ids[-1]) - 1, max_len - 1, replace=False)
                idx = idx + 1
                idx = np.insert(idx, 0, 0)
            gene_ids = gene_ids[:, idx]
            values = values[:, idx]
            masked_value = masked_value[:, idx]
        elif len(gene_ids[-1]) < max_len:
            pad_id = self.vocab['<pad>']
            pad_value = -2
            gene_ids = torch.cat([gene_ids, torch.full((1, max_len - gene_ids.size(-1)), pad_id, dtype=gene_ids.dtype)],
                                 dim=-1)
            values = torch.cat([values, torch.full((1, max_len - values.size(-1)), pad_value, dtype=values.dtype)],
                               dim=-1)
            masked_value = torch.cat(
                [masked_value, torch.full((1, max_len - masked_value.size(-1)), pad_value, dtype=masked_value.dtype)],
                dim=-1)
        return values.squeeze(), gene_ids.squeeze(), masked_value.squeeze()

    def _digitize(self, x: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """
        Digitize the data into bins. This method spreads data uniformly when bins
        have same values.

        Args:

        x (:class:`np.ndarray`):
            The data to digitize.
        bins (:class:`np.ndarray`):
            The bins to use for digitization, in increasing order.

        Returns:

        :class:`np.ndarray`:
            The digitized data.
        """
        assert x.ndim == 1 and bins.ndim == 1

        left_digits = np.digitize(x, bins)
        right_difits = np.digitize(x, bins, right=True)

        rands = np.random.rand(len(x))  # uniform random numbers

        digits = rands * (right_difits - left_digits) + left_digits
        digits = np.ceil(digits).astype(np.int64)
        return digits


def random_mask_value(
        values: Union[torch.Tensor, np.ndarray],
        mask_ratio: float = 0.15,
        mask_value: int = -1,
        pad_value: int = -2,
) -> torch.Tensor:
    """
    Randomly mask a batch of data.

    Args:
        values (array-like):
            A batch of tokenized data, with shape (batch_size, n_features).
        mask_ratio (float): The ratio of genes to mask, default to 0.15.
        mask_value (int): The value to mask with, default to -1.
        pad_value (int): The value of padding in the values, will be kept unchanged.

    Returns:
        torch.Tensor: A tensor of masked data.
    """
    if isinstance(values, torch.Tensor):
        # it is crutial to clone the tensor, otherwise it changes the original tensor
        values = values.clone().detach().numpy()
    else:
        values = values.copy()
    row = values
    if mask_ratio > 0:
        non_padding_idx = np.nonzero(row - pad_value)[0]
        n_mask = int(len(non_padding_idx) * mask_ratio)
        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
        row[mask_idx] = mask_value
    return row


# data_loader

class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def filter_gene(vocab, adata, is_master, logger):
    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var_names.tolist()
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    if is_master:
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]
    return adata, gene_ids_in_vocab


def prepare_cell_data(adata, adata_test, args, vocab,
                      is_master=False, mask_value=-1, pad_value=-2, logger=None, sort_seq_batch=False, pad_token='<pad>'
                      ):
    '''
    Args:
        adata: adata used for training
        adata_test: adata used for testing
        args:
        vocab:
        is_master: does the current GPU act as the master
        mask_value: specify certain values used as mask value (default: -1)
        pad_value: specify certain values used as padding value (default: -2)
        logger:
        sort_seq_batch:
        pad_token:
    Returns:
    '''
    input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
        "normed_raw": "X_normed",
        "log1p": "X_normed",
        "binned": "X_binned",
    }[args.input_style]

    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    genes = adata.var["gene_name"].tolist()

    if args.task in ['Integration', 'Cell_annotation']:
        celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
        celltypes_labels = np.array(celltypes_labels)
        batch_ids = adata.obs["batch_id"].tolist()
        num_batch_types = len(set(batch_ids))
        batch_ids = np.array(batch_ids)
        (
            train_data,
            valid_data,
            train_celltype_labels,
            valid_celltype_labels,
            train_batch_labels,
            valid_batch_labels,
        ) = train_test_split(
            all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
        )
    else:
        (
            train_data,
            valid_data,
        ) = train_test_split(
            all_counts, test_size=0.1, shuffle=True
        )
        num_batch_types = 0

    gene_ids = np.array(vocab(genes), dtype=int)
    if args.graph_sort:
        graph = dgl.load_graphs(args.graph_path)[0][0]
    else:
        graph = None
    random_sort = args.random_sort if 'random_sort' in args else False


    tokenized_train = tokenize_and_pad_batch_v2(
        train_data,
        gene_ids,
        max_len=args.max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=args.append_cls,  # append <cls> token at the beginning
        include_zero_gene=args.include_zero_gene,
        graph=graph,
        random_sort=random_sort,
    )

    tokenized_valid = tokenize_and_pad_batch_v2(
        valid_data,
        gene_ids,
        max_len=args.max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=args.append_cls,
        include_zero_gene=args.include_zero_gene,
        graph=graph
    )

    if is_master:
        logger.info(
            f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
            f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
        )
        logger.info(
            f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
            f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
        )

    masked_values_train = torch.from_numpy(random_mask_value(
        tokenized_train["values"],
        mask_ratio=args.mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )).float()
    masked_values_valid = torch.from_numpy(random_mask_value(
        tokenized_valid["values"],
        mask_ratio=args.mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )).float()

    if adata_test is not None:
        all_counts_test = (
            adata_test.layers[input_layer_key].A
            if issparse(adata_test.layers[input_layer_key])
            else adata_test.layers[input_layer_key]
        )
        if args.task in ['Integration', 'Cell_annotation']:
            celltypes_labels_test = adata_test.obs["celltype_id"].tolist()  # make sure count from 0
            celltypes_labels_test = np.array(celltypes_labels_test)
            batch_ids_test = adata_test.obs["batch_id"].tolist()
            batch_ids_test = np.array(batch_ids_test)

        tokenized_test = tokenize_and_pad_batch_v2(
            all_counts_test,
            gene_ids,
            max_len=args.max_seq_len,
            vocab=vocab,
            pad_token=pad_token,
            pad_value=pad_value,
            append_cls=args.append_cls,  # append <cls> token at the beginning
            include_zero_gene=args.include_zero_gene,
            graph=graph
        )

        input_values_test = torch.from_numpy(random_mask_value(
            tokenized_test["values"],
            mask_ratio=args.mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )).float()
        test_data_pt = {
            "gene_ids": tokenized_test["genes"],
            "values": input_values_test,
            "target_values": tokenized_test["values"],
            "sorted_layer_idx": tokenized_test["sorted_layer_idx"]
        }
        if args.task in ['Integration', 'Cell_annotation']:
            test_data_pt.update({"batch_labels": torch.from_numpy(batch_ids_test).long(),
                                 "celltype_labels": torch.from_numpy(celltypes_labels_test).long()})

    else:
        test_data_pt = None

    if is_master:
        print(
            f"Ratio of masked values in train: ",
            f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
        )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )
    if args.task in ['Integration', 'Cell_annotation']:
        tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
        tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()
        tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
        tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

    if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "sorted_layer_idx": tokenized_train["sorted_layer_idx"]
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "sorted_layer_idx": tokenized_valid["sorted_layer_idx"]
    }
    if args.task in ['Integration', 'Cell_annotation']:
        train_data_pt.update({"batch_labels": tensor_batch_labels_train,
                              "celltype_labels": tensor_celltype_labels_train, })
        valid_data_pt.update({"batch_labels": tensor_batch_labels_valid,
                              "celltype_labels": tensor_celltype_labels_valid, })

    return train_data_pt, valid_data_pt, test_data_pt, num_batch_types


def prepare_cell_data_for_infer(adata,
                                args,
                                vocab,
                                is_master=True,
                                mask_value=-1,
                                pad_value=-2,
                                logger=None,
                                sort_seq_batch=False,
                                pad_token='<pad>'
                                ):
    '''
    Args:
        adata: adata used for training
        args:
        vocab:
        is_master: does the current GPU act as the master
        mask_value: specify certain values used as mask value (default: -1)anno_1_2k_run_ms.out.284279
        pad_value: specify certain values used as padding value (default: -2)
        logger:
        sort_seq_batch:
        pad_token:
    Returns:
    '''
    input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
        "normed_raw": "X_normed",
        "log1p": "X_normed",
        "binned": "X_binned",
    }[args.input_style]

    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels = np.array(celltypes_labels)
    batch_ids = adata.obs["batch_id"].tolist()
    num_batch_types = len(set(batch_ids))
    batch_ids = np.array(batch_ids)
    genes = adata.var["gene_name"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)
    if args.graph_sort:
        graph = dgl.load_graphs(args.graph_path)[0][0]
    else:
        graph = None
    random_sort = args.random_sort if 'random_sort' in args else False
    tokenized_train = tokenize_and_pad_batch_v2(
        all_counts,
        gene_ids,
        max_len=args.max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=args.append_cls,  # append <cls> token at the beginning
        include_zero_gene=args.include_zero_gene,
        graph=graph,
        random_sort=random_sort

    )
    if is_master:
        logger.info(
            f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
            f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
        )
    masked_values_train = torch.from_numpy(random_mask_value(
        tokenized_train["values"],
        mask_ratio=args.mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )).float()
    if is_master:
        print(
            f"Ratio of masked values in train: ",
            f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
        )
    input_gene_ids_train = tokenized_train["genes"]
    input_values_train = masked_values_train
    target_values_train = tokenized_train["values"]
    tensor_batch_labels_train = torch.from_numpy(batch_ids).long()
    tensor_celltype_labels_train = torch.from_numpy(celltypes_labels).long()
    if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
        train_sort_ids = np.argsort(celltypes_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "sorted_layer_idx": tokenized_train["sorted_layer_idx"],
        "batch_labels": tensor_batch_labels_train,
        "celltype_labels": tensor_celltype_labels_train
    }
    return train_data_pt

def compute_nonzero_len_quantile(adata, layer_key="X_binned", q=0.9):
    X = adata.layers[layer_key]
    X = X.toarray() if hasattr(X, "toarray") else X
    nonzero_counts = (X > 0).sum(axis=1)
    return np.quantile(nonzero_counts, q)