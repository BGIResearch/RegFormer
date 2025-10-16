import json
import pickle
from pathlib import Path
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple, Union
from typing_extensions import Self

import numpy as np
import pandas as pd
import dgl
import torch
import torchtext.vocab as torch_vocab
from torchtext.vocab import Vocab
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import AutoTokenizer, BertTokenizer

from regformer import logger


class GeneTokenizer(PreTrainedTokenizer):
    pass


class GeneVocab(Vocab):
    """
    Vocabulary for genes.
    """

    def __init__(
        self,
        gene_list_or_vocab: Union[List[str], Vocab],
        specials: Optional[List[str]] = None,
        special_first: bool = True,
        default_token: Optional[str] = "<pad>",
    ) -> None:
        """
        Initialize the vocabulary.
        Note: add specials only works when init from a gene list.

        Args:
            gene_list_or_vocab (List[str] or Vocab): List of gene names or a
                Vocab object.
            specials (List[str]): List of special tokens.
            special_first (bool): Whether to add special tokens to the beginning
                of the vocabulary.
            default_token (str): Default token, by default will set to "<pad>",
                if "<pad>" is in the vocabulary.
        """
        if isinstance(gene_list_or_vocab, Vocab):
            _vocab = gene_list_or_vocab
            if specials is not None:
                raise ValueError(
                    "receive non-empty specials when init from a Vocab object."
                )
        elif isinstance(gene_list_or_vocab, list):
            _vocab = self._build_vocab_from_iterator(
                gene_list_or_vocab,
                specials=specials,
                special_first=special_first,
            )
        else:
            raise ValueError(
                "gene_list_or_vocab must be a list of gene names or a Vocab object."
            )
        super().__init__(_vocab.vocab)
        if default_token is not None and default_token in self:
            self.set_default_token(default_token)

    @classmethod
    def from_file(cls, file_path: Union[Path, str]) -> Self:
        """
        Load the vocabulary from a file. The file should be either a pickle or a
        json file of token to index mapping.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if file_path.suffix == ".pkl":
            with file_path.open("rb") as f:
                vocab = pickle.load(f)
                return cls(vocab)
        elif file_path.suffix == ".json":
            with file_path.open("r") as f:
                token2idx = json.load(f)
                return cls.from_dict(token2idx)
        else:
            raise ValueError(
                f"{file_path} is not a valid file type. "
                "Only .pkl and .json are supported."
            )

    @classmethod
    def from_dict(
        cls,
        token2idx: Dict[str, int],
        default_token: Optional[str] = "<pad>",
    ) -> Self:
        """
        Load the vocabulary from a dictionary.

        Args:
            token2idx (Dict[str, int]): Dictionary mapping tokens to indices.
        """
        # initiate an empty vocabulary first
        _vocab = cls([])

        # add the tokens to the vocabulary, GeneVocab requires consecutive indices
        for t, i in sorted(token2idx.items(), key=lambda x: x[1]):
            _vocab.insert_token(t, i)

        if default_token is not None and default_token in _vocab:
            _vocab.set_default_token(default_token)

        return _vocab

    def _build_vocab_from_iterator(
        self,
        iterator: Iterable,
        min_freq: int = 1,
        specials: Optional[List[str]] = None,
        special_first: bool = True,
    ) -> Vocab:
        """
        Build a Vocab from an iterator. This function is modified from
        torchtext.vocab.build_vocab_from_iterator. The original function always
        splits tokens into characters, which is not what we want.

        Args:
            iterator (Iterable): Iterator used to build Vocab. Must yield list
                or iterator of tokens.
            min_freq (int): The minimum frequency needed to include a token in
                the vocabulary.
            specials (List[str]): Special symbols to add. The order of supplied
                tokens will be preserved.
            special_first (bool): Whether to add special tokens to the beginning

        Returns:
            torchtext.vocab.Vocab: A `Vocab` object
        """

        counter = Counter()
        counter.update(iterator)

        if specials is not None:
            for tok in specials:
                del counter[tok]

        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[0])
        sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)

        if specials is not None:
            if special_first:
                specials = specials[::-1]
            for symbol in specials:
                ordered_dict.update({symbol: min_freq})
                ordered_dict.move_to_end(symbol, last=not special_first)

        word_vocab = torch_vocab.vocab(ordered_dict, min_freq=min_freq)
        return word_vocab

    @property
    def pad_token(self) -> Optional[str]:
        """
        Get the pad token.
        """
        if getattr(self, "_pad_token", None) is None:
            self._pad_token = None
        return self._pad_token

    @pad_token.setter
    def pad_token(self, pad_token: str) -> None:
        """
        Set the pad token. Will not add the pad token to the vocabulary.

        Args:
            pad_token (str): Pad token, should be in the vocabulary.
        """
        if pad_token not in self:
            raise ValueError(f"{pad_token} is not in the vocabulary.")
        self._pad_token = pad_token

    def save_json(self, file_path: Union[Path, str]) -> None:
        """
        Save the vocabulary to a json file.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        with file_path.open("w") as f:
            json.dump(self.get_stoi(), f, indent=2)

    def set_default_token(self, default_token: str) -> None:
        """
        Set the default token.

        Args:
            default_token (str): Default token.
        """
        if default_token not in self:
            raise ValueError(f"{default_token} is not in the vocabulary.")
        self.set_default_index(self[default_token])


def get_default_gene_vocab() -> GeneVocab:
    """
    Get the default gene vocabulary, consisting of gene symbols and ids.
    """
    vocab_file = Path(__file__).parent / "default_gene_vocab.json"
    if not vocab_file.exists():
        logger.info(
            f"No existing default vocab, will build one and save to {vocab_file}"
        )
        return _build_default_gene_vocab(save_vocab_to=vocab_file)
    logger.info(f"Loading gene vocabulary from {vocab_file}")
    return GeneVocab.from_file(vocab_file)


def _build_default_gene_vocab(
    download_source_to: str = "/tmp",
    save_vocab_to: Union[Path, str, None] = None,
) -> GeneVocab:
    """
    Build the default gene vocabulary from HGNC gene symbols.

    Args:
        download_source_to (str): Directory to download the source data.
        save_vocab_to (Path or str): Path to save the vocabulary. If None,
            the vocabulary will not be saved. Default to None.
    """
    gene_collection_file = (
        Path(download_source_to) / "human.gene_name_symbol.from_genenames.org.tsv"
    )
    if not gene_collection_file.exists():
        # download and save file from url
        url = (
            "https://www.genenames.org/cgi-bin/download/custom?col=gd_app_sym&"
            "col=md_ensembl_id&status=Approved&status=Entry%20Withdrawn&hgnc_dbtag"
            "=on&order_by=gd_app_sym_sort&format=text&submit=submit"
        )
        import requests

        r = requests.get(url)
        gene_collection_file.write_text(r.text)

    logger.info(f"Building gene vocabulary from {gene_collection_file}")
    df = pd.read_csv(gene_collection_file, sep="\t")
    gene_list = df["Approved symbol"].dropna().unique().tolist()
    gene_vocab = GeneVocab(gene_list)  # no special tokens set in default vocab
    if save_vocab_to is not None:
        gene_vocab.save_json(Path(save_vocab_to))
    return gene_vocab


def tokenize_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    return_pt: bool = True,
    include_zero_gene: bool = False,
    graph=None,
    random_sort=False
) -> List[Tuple[Union[torch.Tensor, np.ndarray]]]:
    """
    Tokenize a batch of data. Returns a list of tuple (gene_id, count).

    Args:
        data (array-like): A batch of data, with shape (batch_size, n_features).
            n_features equals the number of all genes.
        gene_ids (array-like): A batch of gene ids, with shape (n_features,).
        return_pt (bool): Whether to return torch tensors of gene_ids and counts,
            default to True.

    Returns:
        list: A list of tuple (gene_id, count) of non zero gene expressions.
    """
    if data.shape[1] != len(gene_ids):
        raise ValueError(
            f"Number of features in data ({data.shape[1]}) does not match "
            f"number of gene_ids ({len(gene_ids)})."
        )
    tokenized_data = []
    for i in range(len(data)):
        row = data[i]
        if include_zero_gene:
            values = row
            genes = gene_ids
        else:
            idx = np.nonzero(row)[0]
            values = row[idx]
            genes = gene_ids[idx]
        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values)
        if graph is not None:
            if random_sort:
                genes, sort_layer_idx, values = random_sorting(genes, values)
                if return_pt:
                    genes = torch.from_numpy(genes).long()
                    sort_layer_idx = torch.from_numpy(sort_layer_idx).long()
                    values = torch.from_numpy(values)
            else:
                grn = dgl.node_subgraph(graph, genes)
                topo_sorting=dgl.topological_nodes_generator(grn)
                sorted_index = torch.cat(topo_sorting)
                sort_layer_idx = []
                for idx, layer in enumerate(topo_sorting):
                    sort_layer_idx += [idx + 1] * len(layer)
                sort_layer_idx=torch.tensor(sort_layer_idx,dtype=torch.int64)
                genes=grn.ndata['_ID'][sorted_index]
                values=values[sorted_index]
        else:
            sort_layer_idx=torch.zeros_like(torch.from_numpy(gene_ids)) if return_pt else np.zeros_like(gene_ids)


        if not return_pt:
            genes=genes.numpy()
            values = values.numpy()
            sort_layer_idx = sort_layer_idx.numpy()

        tokenized_data.append((genes, values,sort_layer_idx))

    return tokenized_data


def random_sorting(gene_ids, values):
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


def pad_batch(
    batch: List[Tuple],
    max_len: int,
    vocab: Vocab,
    pad_token: str = "<pad>",
    pad_value: int = 0,
    append_cls: bool = True,
    cls_id: int = "<cls>",
    graph=None
) -> Dict[str, torch.Tensor]:
    """
    Pad a batch of data. Returns a list of Dict[gene_id, count].

    Args:
        batch (list): A list of tuple (gene_id, count).
        max_len (int): The maximum length of the batch.
        vocab (Vocab): The vocabulary containing the pad token.
        pad_token (str): The token to pad with.

    Returns:
        Dict[str, torch.Tensor]: A dictionary of gene_id and count.
    """
    pad_id = vocab[pad_token]
    gene_ids_list = []
    values_list = []
    sorted_layer_idx_list=[]
    for i in range(len(batch)):
        gene_ids, values ,sorted_layer_idx= batch[i]
        if len(gene_ids) > max_len:
            # sample max_len genes
            idx = np.random.choice(len(gene_ids), max_len, replace=False)
            gene_ids = gene_ids[idx]
            values = values[idx]
            if graph is not None:
                sorted_layer_idx=sorted_layer_idx[idx]
            else:
                sorted_layer_idx=torch.zeros_like(gene_ids)
        if len(gene_ids) < max_len:
            gene_ids = torch.cat(
                [
                    gene_ids,
                    torch.full(
                        (max_len - len(gene_ids),), pad_id, dtype=gene_ids.dtype
                    ),
                ]
            )
            values = torch.cat(
                [
                    values,
                    torch.full((max_len - len(values),), pad_value, dtype=values.dtype),
                ]
            )
            if graph is not None:
                sorted_layer_idx=torch.cat(
                [
                    sorted_layer_idx,
                    torch.full((max_len - len(sorted_layer_idx),), pad_value, dtype=sorted_layer_idx.dtype),
                ]
            )
            else:
                sorted_layer_idx=torch.zeros_like(gene_ids)

        if append_cls:
            gene_ids = torch.cat([gene_ids, torch.tensor([cls_id], dtype=gene_ids.dtype)])
            values = torch.cat([values, torch.tensor([0], dtype=values.dtype)])
            sorted_layer_idx = torch.cat([sorted_layer_idx, torch.tensor([0], dtype=sorted_layer_idx.dtype)])
        gene_ids_list.append(gene_ids)
        values_list.append(values)
        sorted_layer_idx_list.append(sorted_layer_idx)
    batch_padded = {
        "genes": torch.stack(gene_ids_list, dim=0),
        "values": torch.stack(values_list, dim=0),
        "sorted_layer_idx":torch.stack(sorted_layer_idx_list, dim=0),

    }
    return batch_padded


def tokenize_and_pad_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    max_len: int,
    vocab: Vocab,
    pad_token: str,
    pad_value: int,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_token: str = "<cls>",
    return_pt: bool = True,
    graph=None,
    random_sort=False,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize and pad a batch of data. Returns a list of tuple (gene_id, count).
    """
    cls_id = vocab[cls_token]
    tokenized_data = tokenize_batch(
        data,
        gene_ids,
        return_pt=return_pt,
        include_zero_gene=include_zero_gene,
        graph=graph,
        random_sort=random_sort
    )
    if include_zero_gene:
        max_len=gene_ids.__len__()+1 if append_cls else gene_ids.__len__()
    batch_padded = pad_batch(
        tokenized_data, max_len, vocab, pad_token, pad_value, append_cls=append_cls, cls_id=cls_id, graph=graph
    )
    return batch_padded


def random_mask_value(
    values: Union[torch.Tensor, np.ndarray],
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    pad_value: int = 0,
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

    for i in range(len(values)):
        row = values[i]
        non_padding_idx = np.nonzero(row - pad_value)[0]
        n_mask = int(len(non_padding_idx) * mask_ratio)
        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
        row[mask_idx] = mask_value
    return torch.from_numpy(values).float()


def tokenize_and_pad_batch_v2(
    data: np.ndarray,
    gene_ids: np.ndarray,
    max_len: int,
    vocab: Vocab,
    pad_token: str,
    pad_value: int,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_token: str = "<cls>",
    graph=None,
    random_sort=False,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize → truncate → graph sort → pad → append CLS (no mask)

    Args:
        data (np.ndarray): Expression matrix of shape (batch_size, n_features).
        gene_ids (np.ndarray): Gene IDs of shape (n_features,).
        max_len (int): Maximum number of tokens (excluding CLS token).
        vocab (Vocab): Token-to-ID vocabulary dictionary.
        pad_token (str): Padding token (e.g., "<pad>").
        pad_value (int): Padding value for expression counts.
        append_cls (bool): Whether to append CLS token at the end.
        include_zero_gene (bool): If True, include zero-expression genes.
        cls_token (str): CLS token string (default "<cls>").
        graph (DGLGraph): Optional graph for topological sorting.
        random_sort (bool): If True, apply random sorting over graph topology.

    Returns:
        Dict[str, torch.Tensor]: A dictionary with keys:
            - 'genes': (batch_size, seq_len)
            - 'values': (batch_size, seq_len)
            - 'sorted_layer_idx': (batch_size, seq_len)
            where seq_len = max_len (+1 if append_cls=True)
    """
    cls_id = vocab[cls_token]
    pad_id = vocab[pad_token]

    batch_gene_ids = []
    batch_values = []
    batch_sorted_layer_idx = []

    for i in range(len(data)):
        row = data[i]

        # Step 1: Tokenize (select non-zero genes if not include_zero_gene)
        if include_zero_gene:
            values = row
            genes = gene_ids
        else:
            idx = np.nonzero(row)[0]
            values = row[idx]
            genes = gene_ids[idx]

        # Step 2: Truncate if longer than max_len
        if len(genes) > max_len:
            idx = np.random.choice(len(genes), max_len, replace=False)
            genes = genes[idx]
            values = values[idx]

        # Step 3: Graph-based sorting (topological or random)
        if graph is not None:
            if random_sort:
                genes, sort_layer_idx, values = random_sorting(genes, values)
            else:
                grn = dgl.node_subgraph(graph, torch.tensor(genes).long())
                topo_sorting = dgl.topological_nodes_generator(grn)
                sorted_index = torch.cat(topo_sorting)

                # Create sorted_layer_idx: layer number for each gene token
                sort_layer_idx = []
                for idx_layer, layer in enumerate(topo_sorting):
                    sort_layer_idx += [idx_layer + 1] * len(layer)
                sort_layer_idx = np.array(sort_layer_idx, dtype=np.int64)

                # Apply sorting
                genes = grn.ndata["_ID"][sorted_index].numpy()
                values = values[sorted_index.numpy()]
        else:
            sort_layer_idx = np.zeros(len(genes), dtype=np.int64)

        # Step 4: Padding (after sorting)
        def pad(arr, target_len, pad_val):
            if len(arr) < target_len:
                arr = np.concatenate(
                    [arr, np.full(target_len - len(arr), pad_val, dtype=arr.dtype)]
                )
            return arr

        genes = pad(genes, max_len, pad_id)
        values = pad(values, max_len, pad_value)
        sort_layer_idx = pad(sort_layer_idx, max_len, 0)

        # Step 5: Append CLS token (optional)
        if append_cls:
            genes = np.insert(genes, max_len, cls_id)
            values = np.insert(values, max_len, 0)
            sort_layer_idx = np.insert(sort_layer_idx, max_len, 0)

        # Step 6: Convert to tensor and collect
        batch_gene_ids.append(torch.tensor(genes).long())
        batch_values.append(torch.tensor(values).float())
        batch_sorted_layer_idx.append(torch.tensor(sort_layer_idx).long())

    # Stack into batched tensors
    batch = {
        "genes": torch.stack(batch_gene_ids, dim=0),
        "values": torch.stack(batch_values, dim=0),
        "sorted_layer_idx": torch.stack(batch_sorted_layer_idx, dim=0),
    }

    return batch
