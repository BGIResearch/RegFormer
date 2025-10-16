# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:inference_utils.py
# @Software:PyCharm
# @Created Time:2024/5/9 10:33 AM
import numpy as np
import pandas as pd


def remove_gene(X, vocab,gene_list=None):    # remove gene not in vocab
    gene_vocab = vocab.vocab.itos_
    # vocab list
    if not gene_list:
        gene_list = X.columns.tolist()   # gene list
    valid_idx = []     # True / False
    gene_ids = []      # extract gene ids in vocab
    for gene in gene_list:
        if gene in gene_vocab:
            valid_idx.append(True)
            gene_ids.append(vocab[gene])
        else:
            valid_idx.append(False)
    if isinstance(X,pd.DataFrame):
        X = X.loc[:, valid_idx]      # filter with True / False
    else:
        X=X[:,valid_idx]
    gene_name=np.array(gene_list)[valid_idx]
    #assert all(gene in gene_list for gene in X_df)
    #assert all(gene in gene_vocab for gene in X_df)
    return X, gene_ids,gene_name


def filter_expressed_gene(pretrain_gene_x, gene_ids):   # keep expressed gene only
    mask = pretrain_gene_x > 0
    filtered_gene = pretrain_gene_x[mask]
    filtered_gene_ids = gene_ids[mask]
    return filtered_gene, filtered_gene_ids


def digitize(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
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
    right_digits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))  # uniform random numbers

    digits = rands * (right_digits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits

def binning(values,n_bins):
    non_zero_ids = values.nonzero()
    non_zero_row = values[non_zero_ids]
    bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
    # bins = np.sort(np.unique(bins))
    # NOTE: comment this line for now, since this will make the each category
    # has different relative meaning across datasets
    non_zero_digits = digitize(non_zero_row, bins)
    assert non_zero_digits.min() >= 1
    assert non_zero_digits.max() <= n_bins - 1
    binned_row = np.zeros_like(values, dtype=np.int64).copy()
    binned_row[non_zero_ids] = non_zero_digits
    bin_edge = np.concatenate([[0], bins])
    return binned_row, bin_edge


