#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: lmdb_writer.py
@time: 2024/11/6 16:03
"""
import os
import lmdb
import scanpy as sc
import json
from scipy.sparse import issparse
import numpy as np
from traceback import print_exc
from pathlib import Path
from datetime import datetime


class LmdbHelper(object):
    def __init__(self,
                 lmdb_path,
                 vocab_path,
                 files):
        self.lmdb_path = lmdb_path
        with open(vocab_path, 'r') as fd:
            self.gene_vocab = json.load(fd)
        self.db = None
        self.txn = None
        self.files = files
        self.file_index = self.get_file_index(files)

    @staticmethod
    def get_cells_number(adata_path):
        adata = sc.read_h5ad(adata_path, backed='r')
        cells = adata.obs.shape[0]
        adata.file.close()
        return cells

    def get_file_index(self, files):
        file_index = {}
        for file_path in files:
            file_index[file_path] = Path(file_path).name.split('.h5ad')[0]
        return file_index

    def get_txn(self, write=True):
        return self.db.begin(write=write)

    def init_db(self, map_size, is_write):
        self.db = lmdb.open(self.lmdb_path, map_size=map_size, sync=False)

    def get_length(self):
        res = self.txn.get(b'__len__')
        length = int(res.decode("utf-8")) if res else 0
        return length

    def h5ad2lmdb(self, h5ad_file, db_length=0):
        cells = self.get_cells_number(h5ad_file)
        print('cell number: {}'.format(cells))
        gene_index = self.file_index[h5ad_file]
        adata = sc.read_h5ad(h5ad_file)
        db_length = self.parse_cxg_data(adata, gene_index, db_length)
        return db_length

    def folder2lmdb(self, db_length=0):
        h5ad_folder = os.path.dirname(self.files[0])
        if os.path.exists(self.lmdb_path + '/deal_files.txt'):
            with open(self.lmdb_path + '/deal_files.txt', 'r') as tmpf:
                deal_file_list = [i.strip('\n') for i in tmpf]
        else:
            deal_file_list = []
        fd = open(self.lmdb_path + '/deal_files.txt', 'a')
        for f in self.files:
            if len(deal_file_list) > 0 and os.path.join(h5ad_folder, f) in deal_file_list:
                continue
            try:
                print("db_length: {}".format(db_length))
                if f.endswith('.h5ad'):
                    db_length = self.h5ad2lmdb(os.path.join(h5ad_folder, f), db_length)
                fd.write('{}\n'.format(os.path.join(h5ad_folder, f)))
                fd.flush()
                print("end to deal file: {}".format(os.path.join(h5ad_folder, f)))
            except Exception as e:
                print("error: {}, file: {}".format(e, os.path.join(h5ad_folder, f)))
                print_exc()
        # self.txn.commit()
        # self.txn = self.get_txn(write=True)
        print('end to load data {}, db_length: {}'.format(h5ad_folder, db_length))
        self.db.sync()
        self.db.close()
        return db_length

    def get_token_ids(self, tokens_list):
        token_ids = []
        use_index = []
        for k, i in enumerate(tokens_list):
            if i in self.gene_vocab:
                use_index.append(k)
                token_ids.append(self.gene_vocab[i])
        return token_ids, use_index

    @staticmethod
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

    def parse_cxg_data(self, adata, gene_index, db_length=0):
        # adata = self.data_preprocess(adata, 'feature_name', self.gene_vocab.keys())
        if adata is not None:
            print(f'cell number after sample: {adata.shape[0]}')
            sparse_flag = issparse(adata.X)

            gene_id, _ = self.get_token_ids(adata.var['feature_name'].values)
            adata.X = adata.X.astype(float)
            print('gene index: {} '.format(gene_index))
            buffer = []
            buffer_len = []
            for i in range(adata.obs.shape[0]):
                express_x = adata.X[i].A.reshape(-1).tolist() if sparse_flag else list(adata.X[i])
                if len(express_x) != len(gene_id):
                    print('data Error: the length of genes is not equal to the length of express_x.')
                    continue
                result = {}
                result['express_x'] = express_x
                result['sex_ontology_term_id'] = adata.obs['sex_ontology_term_id'].iloc[i]
                result['suspension_type'] = adata.obs['suspension_type'].iloc[i]
                result['assay_ontology_term_id'] = adata.obs['assay_ontology_term_id'].iloc[i]
                result['cell_type_ontology_term_id'] = adata.obs['cell_type_ontology_term_id'].iloc[i]
                result['development_stage_ontology_term_id'] = adata.obs['development_stage_ontology_term_id'].iloc[i]
                result['disease_ontology_term_id'] = adata.obs['disease_ontology_term_id'].iloc[i]
                result['tissue_ontology_term_id'] = adata.obs['tissue_ontology_term_id'].iloc[i]
                result['celltype'] = adata.obs['cell_type'].iloc[i]
                result['gene_index'] = 'g{}'.format(gene_index)
                buffer.append(result)
                buffer_len.append(db_length)
                db_length += 1
                batch_size = 5000
                if len(buffer) >= batch_size:
                    self._write_batch(buffer, buffer_len)
                    buffer.clear()
                    buffer_len.clear()
            if buffer:
                self._write_batch(buffer, buffer_len)
                buffer.clear()
                buffer_len.clear()
            with self.db.begin(write=True) as txn:
                txn.put(b'__len__', str(db_length).encode())
                txn.put('g{}'.format(gene_index).encode(), np.array(gene_id))
        return db_length

    def _write_batch(self, buffer, buffer_len):
        with self.db.begin(write=True) as txn:
            for i in range(len(buffer)):
                txn.put(str(buffer_len[i]).encode(), json.dumps(buffer[i]).encode())
                db_length = buffer_len[i]
            print(f'{datetime.now().strftime("%y-%m-%d %H:%M:%S")} write to lmdb: {db_length}')

    def write_lmdb(self, record, db_length):
        index = db_length
        self.txn.put(str(index).encode(), json.dumps(record).encode())
        return index

    def read_lmdb(self, index):
        res = json.loads(self.txn.get(str(index).encode()))
        gene_id = np.frombuffer(self.txn.get(res['gene_index']), dtype=np.int64)
        res['gene_id'] = gene_id
        return res


if __name__ == '__main__':
    import pandas as pd
    import os

    task = 'train'
    vocab_path = '/home/share/huadjyin/home/s_qiuping1/workspace/RegFormer/example/data/vocab.json'
    map_size = int(1024 * 1024 * 1024 * 1024)  # 1T

    if task == 'train':
        data_path = '/home/share/huadjyin/home/s_qiuping1/workspace/RegFormer/example/data/train/'
        files = [data_path + f for f in os.listdir(data_path)]
        lmdb_path = '/home/share/huadjyin/home/s_qiuping1/workspace/RegFormer/example/data/lmdb/train_114w.db'

    if task == 'val':
        data_path = '/home/share/huadjyin/home/s_qiuping1/workspace/RegFormer/example/data/val/'
        files = [data_path + f for f in os.listdir(data_path)]
        lmdb_path = '/home/share/huadjyin/home/s_qiuping1/workspace/RegFormer/example/data/lmdb/val.db'

    obj = LmdbHelper(lmdb_path,
                     vocab_path,
                     files
                     )

    init_num = 0
    print('lmbd length is: {}'.format(init_num))
    obj.init_db(map_size, is_write=True)
    db_len = obj.folder2lmdb(init_num)
    print(db_len)
