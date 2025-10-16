from typing import Dict, Optional, Union

import numpy as np
from scipy.sparse import issparse
import scanpy as sc
from scanpy.get import _get_obs_rep
from anndata import AnnData

from regformer import logger

from typing import Dict

from typing import Dict, Optional, Union

import numpy as np
from scipy.sparse import issparse
import scanpy as sc
from anndata import AnnData

from regformer import logger


def _get_obs_rep(adata: AnnData, layer: Optional[str] = None) -> np.ndarray:
    if layer is None:
        return adata.X
    else:
        return adata.layers[layer]


def _set_obs_rep(adata: AnnData, data: np.ndarray, layer: Optional[str] = None):
    if layer is None:
        adata.X = data
    else:
        adata.layers[layer] = data


class Preprocessor:
    def __init__(
        self,
        use_key: Optional[str] = None,
        filter_gene_by_counts: Union[int, bool] = False,
        filter_cell_by_counts: Union[int, bool] = False,
        normalize_total: Union[float, bool] = 1e4,
        result_normed_key: Optional[str] = None,
        log1p: bool = False,
        result_log1p_key: Optional[str] = None,
        subset_hvg: Union[int, bool] = False,
        use_cell_type: Optional[bool] = False,
        hvg_use_key: Optional[str] = None,
        hvg_flavor: str = "seurat_v3",
        binning: Optional[int] = None,
        result_binned_key: str = "X_binned",
        verbose: bool = True,
    ):
        self.use_key = use_key
        self.filter_gene_by_counts = filter_gene_by_counts
        self.filter_cell_by_counts = filter_cell_by_counts
        self.normalize_total = normalize_total
        self.result_normed_key = result_normed_key
        self.log1p = log1p
        self.result_log1p_key = result_log1p_key
        self.subset_hvg = subset_hvg
        self.use_cell_type = use_cell_type
        self.hvg_use_key = hvg_use_key
        self.hvg_flavor = hvg_flavor
        self.binning = binning
        self.result_binned_key = result_binned_key
        self.verbose = verbose

    def __call__(self, adata: AnnData, batch_key: Optional[str] = None) -> AnnData:
        key = self.use_key if self.use_key != "X" else None
        is_logged = self.check_logged(adata, key)

        # ------------------- Filter -------------------
        if self.filter_gene_by_counts:
            if self.verbose:
                logger.info("🔧 Filtering genes ...")
            sc.pp.filter_genes(adata, min_counts=self.filter_gene_by_counts)

        if self.filter_cell_by_counts:
            if self.verbose:
                logger.info("🔧 Filtering cells ...")
            sc.pp.filter_cells(adata, min_counts=self.filter_cell_by_counts)

        # ------------------- Normalize -------------------
        if self.normalize_total:
            if self.verbose:
                logger.info("⚙️  Normalizing ...")

            normed = sc.pp.normalize_total(
                adata,
                target_sum=self.normalize_total,
                layer=key,
                inplace=False
            )["X"]

            adata.X = normed
            if self.verbose:
                logger.info("✅ Normalized data saved to adata.X")

        # ------------------- Log1p -------------------
        if self.log1p:
            if self.verbose:
                logger.info("⚙️  Applying log1p ...")

            sc.pp.log1p(adata, layer=key)

        # ------------------- use cell type -------------------
        if self.use_cell_type:

            rng = np.random.default_rng(42)

            counts = adata.obs["celltype"].value_counts()
            min_class_count = counts.min()

            max_cells = min_class_count * 20

            keep_indices = []
            for ct, idx in adata.obs.groupby("celltype").indices.items():
                n = len(idx)
                if n > max_cells:
                    sampled = rng.choice(idx, size=max_cells, replace=False)
                else:
                    sampled = idx
                keep_indices.extend(sampled)

            adata = adata[keep_indices].copy()

        if self.result_normed_key:
            adata.layers[self.result_normed_key] = adata.X
            logger.info(f"✅ Normalized data saved to layers['{self.result_normed_key}']")

        if self.result_log1p_key:
            adata.layers[self.result_log1p_key] = adata.X
            logger.info(f"📝 Pre-log data saved to layers['{self.result_log1p_key}']")

        # ------------------- HVG -------------------
        if self.subset_hvg:
            if self.verbose:
                logger.info("🔍 Selecting HVGs ...")
            sc.pp.highly_variable_genes(
                adata,
                layer=self.hvg_use_key,
                n_top_genes=self.subset_hvg if isinstance(self.subset_hvg, int) else None,
                batch_key=batch_key,
                flavor=self.hvg_flavor,
                subset=True,
            )


        # ------------------- Binning -------------------
        if self.binning:
            if self.verbose:
                logger.info(f"🎲 Binning with {self.binning} bins ...")
            layer_data = _get_obs_rep(adata, layer=key)
            layer_data = layer_data.A if issparse(layer_data) else layer_data

            binned = []
            bin_edges = []
            for row in layer_data:
                non_zero_idx = row.nonzero()[0]
                non_zero_vals = row[non_zero_idx]
                if len(non_zero_vals) == 0:
                    bins = np.zeros(self.binning - 1)
                else:
                    bins = np.quantile(non_zero_vals, np.linspace(0, 1, self.binning - 1))
                digitized = self._digitize(non_zero_vals, bins)
                binned_row = np.zeros_like(row, dtype=np.int64)
                binned_row[non_zero_idx] = digitized
                binned.append(binned_row)
                bin_edges.append(np.concatenate([[0], bins]))

            binned_matrix = np.stack(binned)
            adata.layers[self.result_binned_key] = binned_matrix
            adata.obsm["bin_edges"] = np.stack(bin_edges)

            if self.verbose:
                logger.info(f"✅ Binned data saved to layers['{self.result_binned_key}']")

        return adata

    def _digitize(self, x: np.ndarray, bins: np.ndarray) -> np.ndarray:
        left = np.digitize(x, bins)
        right = np.digitize(x, bins, right=True)
        rands = np.random.rand(len(x))
        digits = np.ceil(rands * (right - left) + left).astype(np.int64)
        return digits

    def check_logged(self, adata: AnnData, obs_key: Optional[str] = None) -> bool:
        data = _get_obs_rep(adata, layer=obs_key)
        if data.min() < 0 or data.max() > 30:
            return False
        non_zero_min = data[data > 0].min() if (data > 0).sum() > 0 else 0
        return non_zero_min < 1
