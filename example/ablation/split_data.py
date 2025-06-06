#!/usr/bin/env python3
# coding: utf-8
"""
@author: seth
@file: split_data
@time: 2025/5/14 16:37

Change Log：
2025/5/14 16:37: create file
"""
from regformer.utils.utils import train_test_split_adata
import numpy as np
import pandas as pd
import scanpy as sc
import os
from regformer.utils.utils import train_test_split_adata


stat_df = pd.read_csv('/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/CFM/examples/result/organs_stat.csv')
df = stat_df[stat_df.species == 'Homo sapiens']
tissue = 'blood'
df = df[df.tissue == tissue]

files = df['file']
target_names = [f.split('/')[-1] for f in files]


def find_files_by_name(root_dir, target_names):
    matched_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename in target_names:
                full_path = os.path.join(dirpath, filename)
                matched_files.append(full_path)
    return matched_files

matched_files = find_files_by_name('/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/cell_gene/data', target_names)


def get_cells_number(adata_path):
    adata = sc.read_h5ad(adata_path, backed='r')
    cells = adata.obs.shape[0]
    adata.file.close()
    return cells

def data_preprocess(adata, var_key, ref_genes, do_log1p=True):
    if adata.raw is not None:
        adata.X = adata.raw.X
    adata_genes = adata.var[var_key].values if len(var_key) > 0 else adata.var_names.values
    _, unique_index = np.unique(adata_genes, return_index=True)
    adata = adata[:, unique_index].copy()
    sc.pp.calculate_qc_metrics(adata)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    if np.min(adata.X) != 0:
        return None
    use_index = []
    adata_genes = adata.var[var_key].values if len(var_key) > 0 else adata.var_names.values
    for k, i in enumerate(adata_genes):
        if i in ref_genes:
            use_index.append(k)
    adata = adata[:, use_index].copy()
    if len(use_index) == 0:
        raise ValueError("No gene names in ref gene, please check the adata.var_names are gene Symbol!")
    print('useful genes: {}'.format(len(use_index)))
    print(adata)
    if adata.X.min() >= 0:
        normalize_total = False
        log1p = False
        if adata.X.max() > 20:
            log1p = True
            if adata.X.max() - np.int32(adata.X.max()) == np.int32(0):
                normalize_total = 1e4
        if not do_log1p:
            log1p = False
        if normalize_total:
            sc.pp.normalize_total(adata, target_sum=normalize_total)
            print("Normalize total...")
        if log1p:
            sc.pp.log1p(adata)
            print("Log1p...")
    else:
        raise Exception('the express matrix have been scale, exit!')
    return adata

def split_data(h5ad_file, output, gene_vocab, tissue=None):
    base_name = os.path.basename(h5ad_file).split('.h5ad')[0]

    cells = get_cells_number(h5ad_file)
    print('cell number before qc: {}'.format(cells))
    if cells > 130000 and cells < 200000:
    #     return 0
    # # if cells > 100000:
    # #     adata = sc.read_h5ad(h5ad_file, backed='r')
    # #     nums = 1
    # #     for j in range(0, cells, 30000):
    # #         sub_adata = adata[j: j + 30000, :].to_memory()
    # #         if tissue is not None:
    # #             sub_adata = sub_adata[(sub_adata.obs['is_primary_data']) & (sub_adata.obs['tissue'] == tissue), :].copy()
    # #         else:
    # #             sub_adata = sub_adata[sub_adata.obs['is_primary_data'], :].copy()
    # #         nums += 1
    # #         print(f'{h5ad_file} cell number: {sub_adata.shape[0]}')
    # #         if sub_adata.shape[0] > 0:
    # #             train_adata, test_adata = train_test_split_adata(sub_adata, test_size=0.2)
    # #             train_adata.write(f'{output}/train/{base_name}_{nums}.h5ad')
    # #             test_adata.write(f'{output}/val/{base_name}_{nums}.h5ad')
    # #             cell_numbers += train_adata.obs.shape[0]
    # #             print(f'end to deal with {h5ad_file} and save to {output}/train/{base_name}_{num}.h5ad')
    # else:
        adata = sc.read_h5ad(h5ad_file)
        adata = data_preprocess(adata, var_key='feature_name', ref_genes=list(gene_vocab.keys()), do_log1p=True)
        if tissue is not None:
            adata = adata[(adata.obs['is_primary_data']) & (adata.obs['tissue'] == tissue), :].copy()
        else:
            adata = adata[adata.obs['is_primary_data'], :].copy()
        # print(f'{h5ad_file} cell number: {adata.shape[0]}')
        if adata.shape[0] > 0:
            train_adata, test_adata = train_test_split_adata(adata, test_size=0.2)
            train_adata.write(f'{output}/train/{base_name}.h5ad')
            test_adata.write(f'{output}/val/{base_name}.h5ad')
            cell_numbers = train_adata.obs.shape[0]
            print(f'number: {cell_numbers}, save to {output}/train/{base_name}.h5ad')
            return cell_numbers
    else:
        return 0



import json

output_dir = '/home/share/huadjyin/home/s_qiuping1/workspace/RegFormer/example/data/'
vocab_path = '/home/share/huadjyin/home/s_qiuping1/workspace/RegFormer/example/data/vocab.json'
with open(vocab_path, 'r') as fd:
    gene_vocab = json.load(fd)
all_numbers = 0
test_files = []
for f in matched_files:
    test_files.append(f)
    try:
        num = split_data(f, output_dir, gene_vocab)
    except Exception as e:
        print('error: {}, file: {}'.format(e, f))
    all_numbers += num
    print('total number: {}'.format(all_numbers))
    if all_numbers > 150000:
        break
print(test_files)
with open('/home/share/huadjyin/home/s_qiuping1/workspace/RegFormer/example/data/test_files.txt', 'w') as f:
    for item in test_files:
        f.write("%s\n" % item)
