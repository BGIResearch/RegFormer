#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/25 15:10
# @Author  : Luni Hu
# @File    : regformer_grn.py
# @Software: PyCharm


import os
import gc
import torch
import shutil
import swanlab
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

import regformer as scmb
from regformer.utils.utils import (
    load_config, seed_all, model_config, load_ckpt
)
from regformer.model.mambaLM import MambaModel
from regformer.data.dataset import infer_dataset, SeqDataset
from regformer.data.gene_tokenizer import GeneVocab
from regformer.utils.utils import aggregate_gene_embeddings

from collections import defaultdict
from gseapy import enrichr
import networkx as nx
from sklearn.cluster import SpectralClustering


class GrnTaskMamba:
    """
    Class to perform Gene Regulatory Network (GRN) analysis using gene embeddings
    derived from a Mamba-based single-cell model (RegFormer/scMamba).
    """

    def __init__(self, config_file, pad_token="<pad>", unk_token="<unk>", cls_token="<cls>"):
        """
        Initializes the GRN task with configuration, logger, tokens, and parameter checks.
        :param config_file: Path to the configuration file.
        :param pad_token: Padding token.
        :param unk_token: Unknown token.
        :param cls_token: Classification token (often treated as unk_token internally).
        """
        self.args = load_config(config_file)  # Load configuration arguments
        self._init_logger()  # Set up the save directory and logger
        self._init_tokens(pad_token, unk_token, cls_token)  # Define special tokens and their numerical values
        self._check_parameters()  # Validate the configuration settings
        self.gene_sets = getattr(self.args, "gene_sets", None)  # Gene sets for enrichment analysis

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

    def _init_tokens(self, pad_token, unk_token, cls_token):
        """
        Defines the names and numerical values for special tokens based on input embedding style.
        """
        self.pad_token = pad_token
        # Determine the unk_token; it acts as <cls> if CLS token is not appended
        self.unk_token = unk_token if self.args.append_cls else "<cls>"
        self.mask_token = "<mask>"
        if self.args.input_emb_style == "category":
            # Numerical values for binned/categorical input
            self.mask_value = self.args.n_bins + 1
            self.pad_value = self.args.n_bins
            self.n_input_bins = self.args.n_bins + 2
        else:
            # Numerical values for continuous input
            self.mask_value, self.pad_value = -1, -2
            self.n_input_bins = self.args.n_bins

    def _check_parameters(self):
        """
        Validates the compatibility of input/output style and embedding style parameters.
        """
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
        Loads model configuration, vocabulary, dataset, DataLoader, and the Mamba model.
        Loads a pre-trained checkpoint if specified.
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

        # Load and preprocess data for inference (cell-level data)
        data_pt = infer_dataset(
            data_path=self.args.data_path,
            args=self.args,
            logger=self.logger,
            vocab=self.vocab,
            is_master=True,
            mask_value=self.mask_value,
            pad_value=self.pad_value,
            pad_token=self.pad_token
        )
        self.cls_count = torch.bincount(data_pt[
                                            'celltype_labels'])  # Calculate class counts (not directly used for GRN but included for consistency)
        dataset = SeqDataset(data_pt)  # Create a sequence dataset
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size,
                                 shuffle=False)  # Create DataLoader for inference

        model = self._load_model(model_configs)  # Instantiate the Mamba model
        if self.args.load_model is not None:
            model = load_ckpt(model, model_file, self.args, self.logger)  # Load pre-trained checkpoint

        return model.to(self.args.device), data_loader

    def _load_model(self, model_configs):
        """
        Instantiates the MambaModel with specific configurations for pre-training or inference.
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
            domain_spec_batchnorm=False,
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

    def get_gene_expression_embedding(self):
        """
        Extracts and aggregates gene embeddings from the trained model across all cells in the dataset.
        :return: gene_emb_dict (dict): {gene_name: aggregated_embedding_vector}
        """
        model, loader = self.load_data_and_model()  # Load model and data
        self.logger.info(f"Start embedding. Total samples: {len(loader.dataset)}")

        gene_embs = {}
        model.eval()  # Set model to evaluation mode

        # Aggregate gene embeddings from all cells/batches
        gene_embeddings, gene_ids = aggregate_gene_embeddings(
            loader,
            model,
            self.args.layer_size,
            vocab=self.vocab,
            pad_token=self.pad_token,
            device=self.args.device
        )
        gene_emb_array = gene_embeddings[gene_ids].cpu().numpy()  # Get array of embeddings
        gene_names = [self.vocab.lookup_token(i) for i in gene_ids]  # Get corresponding gene names

        # Save embeddings and gene names
        np.save(self.save_dir / "gene_embedding.npy", gene_emb_array)
        pd.Series(gene_names).to_csv(self.save_dir / "gene_names.csv", index=False)

        gene_emb_dict = dict(zip(gene_names, gene_emb_array))  # Create dictionary mapping gene names to embeddings
        self.logger.info("Gene embedding complete.")
        return gene_emb_dict

    def construct_grn(self, embeddings=None, top_k=20):
        """
        Constructs a Gene Regulatory Network (GRN) based on the top-k cosine similarity
        of gene embeddings, specifically considering only Transcription Factors (TFs) as source nodes.

        Args:
            embeddings (dict): {gene: embedding vector}. If None, will call self.get_gene_expression_embedding().
            top_k (int): Number of top similar genes to connect for each TF.

        Returns:
            networkx.DiGraph: Constructed gene similarity graph (directed from TF to target).
        """
        if embeddings is None:
            self.embeddings = self.get_gene_expression_embedding()
        else:
            self.embeddings = embeddings

        # Load the list of Transcription Factors (TFs)
        self.tf_list = pd.read_csv(self.args.tf_file, header=None, sep="\t")[0].tolist()

        import networkx as nx
        from sklearn.metrics.pairwise import cosine_similarity

        self.logger.info(f"Construct GRN with top-{top_k} connections per gene")

        # Convert embedding dict to matrix for efficient similarity calculation
        genes = list(self.embeddings.keys())
        embedding_matrix = np.stack([self.embeddings[g] for g in genes])

        # Compute cosine similarity between all gene pairs
        similarity_matrix = cosine_similarity(embedding_matrix)
        print(similarity_matrix.shape)  # Print shape of the similarity matrix

        # Build graph
        G = nx.DiGraph()  # Use a Directed Graph (DiGraph)
        # G.add_nodes_from(genes) # Nodes are added implicitly when edges are added

        for i, gene in enumerate(genes):
            # Only consider Transcription Factors as regulators (source nodes)
            if gene not in self.tf_list:
                continue

            # Exclude self-similarity when finding neighbors
            sim_scores = similarity_matrix[i].copy()
            sim_scores[i] = -1  # Set self-similarity to a low value

            # Find indices with similarity greater than a threshold (e.g., 0.2)
            valid_indices = np.where(sim_scores > 0.2)[0]

            # Get top-k neighbors by sorting valid indices
            if len(valid_indices) > top_k:
                # Select top-k indices with the highest similarity scores
                top_indices = valid_indices[np.argsort(sim_scores[valid_indices])[-top_k:]]
            else:
                top_indices = valid_indices

            for j in top_indices:
                neighbor_gene = genes[j]
                weight = similarity_matrix[i, j]

                # Add directed edge from TF (gene) to target (neighbor_gene) with similarity as weight
                if not G.has_edge(gene, neighbor_gene):
                    G.add_edge(gene, neighbor_gene, weight=weight)
        self.logger.info(
            f"Graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
        )

        return G

    import networkx as nx
    import numpy as np
    import pandas as pd
    from collections import defaultdict
    from sklearn.cluster import SpectralClustering

    def evaluate_grn(self, g_nx, gene_names=None):
        """
        Performs gene module identification via Spectral Clustering on the GRN,
        followed by enrichment analysis of the identified modules using gseapy.enrichr.

        Args:
            g_nx (networkx.DiGraph): Gene similarity graph (GRN).
            gene_names (list): Background gene list for enrichment (optional).

        Returns:
            pandas.DataFrame: Full enrichment results across all cluster numbers.
        """
        enrichment_results = []

        # === Convert to adjacency matrix ===
        node_list = list(g_nx.nodes())  # List of nodes in the graph
        # Convert graph to a numpy array (adjacency matrix), using 'weight' for edge values
        A = nx.to_numpy_array(g_nx, nodelist=node_list, weight="weight")

        # === Try clustering with k=5..30 ===
        for n_clusters in range(5, 31, 5):
            self.logger.info(f"Clustering with SpectralClustering, n_clusters={n_clusters}")

            try:
                # Initialize Spectral Clustering
                spectral = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity="precomputed",  # Use the adjacency matrix as the affinity matrix
                    random_state=42,
                    assign_labels="kmeans"  # Use k-means to finalize cluster labels
                )
                membership = spectral.fit_predict(A)  # Perform clustering
            except Exception as e:
                self.logger.warning(f"❌ Spectral clustering failed at k={n_clusters}: {e}")
                continue

            # Build module dictionary {cluster_id: [gene list]}
            modules = defaultdict(list)
            for node, cluster_id in zip(node_list, membership):
                modules[cluster_id].append(node)

            module_count = len(modules)
            enrichment_count = 0
            tmp_results = []

            # Run enrichment for each module
            for module_id, gene_list in modules.items():
                if len(gene_list) == 0:
                    continue

                self.logger.info(f"Module {module_id} at k={n_clusters}, size={len(gene_list)}")

                try:
                    # Perform gene set enrichment analysis using Enrichr
                    enr = enrichr(
                        gene_list=gene_list,
                        gene_sets=self.gene_sets,  # Gene set library from args
                        outdir=None,  # Do not save to disk directly
                        background=gene_names  # Use all genes in the dataset as background
                    )
                    res_df = enr.results.copy()
                    res_df["n_clusters"] = n_clusters
                    res_df["module"] = module_id

                    tmp_results.append(res_df)
                    enrichment_count += len(res_df)  # Count total enrichment rows

                except Exception as e:
                    self.logger.warning(f"⚠️ Enrichment failed for module {module_id}: {e}")
                    continue

            # Process and summarize enrichment results for the current n_clusters
            if tmp_results:
                res_df_all = pd.concat(tmp_results, ignore_index=True)
                enrichment_results.append(res_df_all)

                # Parse Overlap column (e.g. "5/120")
                overlap_split = res_df_all['Overlap'].str.split('/', expand=True).astype(int)
                res_df_all['overlap_hit'] = overlap_split[0]
                res_df_all['overlap_total'] = overlap_split[1]

                # Filter for significance and minimum overlap size
                filtered = res_df_all[
                    (res_df_all["overlap_hit"] >= 5) &
                    (res_df_all["Adjusted P-value"] < 0.05)
                    ]
                significant_count = filtered.shape[0]
            else:
                significant_count = 0

            self.logger.info(
                f"✅ n_clusters={n_clusters} | modules: {module_count}, enrichment: {enrichment_count}, "
                f"significant: {significant_count}"
            )

        # === Final aggregation and saving ===
        if not enrichment_results:
            self.logger.warning("❗ No enrichment results found.")
            return None

        df = pd.concat(enrichment_results, ignore_index=True)  # Aggregate results from all cluster counts

        enrichment_path = self.save_dir / "grn_enrichment.csv"
        df.to_csv(enrichment_path, index=False)  # Save the full enrichment DataFrame
        self.logger.info(f"💾 Enrichment results saved to {enrichment_path}")

        return df

    def run_grn_analysis(self):
        """
        Executes the entire GRN analysis pipeline: get embeddings, construct network, and evaluate.
        """
        embeddings = self.get_gene_expression_embedding()  # Step 1: Extract gene embeddings
        g = self.construct_grn(embeddings)  # Step 2: Construct the GRN

        edges_df = nx.to_pandas_edgelist(g)
        edges_df.to_csv(self.save_dir / "edges.csv", index=False)  # Save the constructed edges

        # Step 3: Evaluate the GRN (clustering and enrichment)
        # Pass gene names from embeddings for background in enrichment
        self.evaluate_grn(g, gene_names=list(embeddings.keys()))

        self.logger.info("GRN analysis complete.")
        gc.collect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()
    task = GrnTaskMamba(args.config_file)
    task.run_grn_analysis()