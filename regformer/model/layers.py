# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:layers.py
# @Software:PyCharm
# @Created Time:2024/5/22 3:15 PM
import torch
from torch import nn, Tensor
import math
from typing import Dict, Mapping, Optional, Any, Union
import numpy as np
import torch.nn.functional as F

from regformer.model.grad_reverse import grad_reverse
from flash_attn.flash_attention import FlashMHA

class TopoLayerEncoding(torch.nn.Module):
    def __init__(self, d_model, max_layer=500):
        super(TopoLayerEncoding, self).__init__()
        pe = torch.zeros(max_layer, d_model)
        position = torch.arange(0, max_layer, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, layer_index):
        x = x + self.pe[layer_index].view(x.size())
        return x



class AdversarialDiscriminator(nn.Module):
    """
    Discriminator for the adversarial training for batch correction.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = False,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class MVCDecoder(nn.Module):
    """
    Decoder for the masked value prediction for cell embeddings.
    Goal: given cell and query genes, predict the original expr value/bin through cell-emb and gene-emb.

    There are actually three ways of making this, all start with gene_embs -> query_vecs,
    and then:
    1. cell_emb x W x query vecs.
       This one makes the most sense, since in the query space, the query look at
       different dimensions of cel_emb and sync them. This one has explicit interaction.
    2. FC([cell_emb, query_vecs]).
       This one has the benifit to have smaller query_vecs and makes them like bottle
       neck layer. For tutorials 64 dims.
    3. FC(cell_emb + query_vecs).

    **NOTE**:
    1. it is important to make gene query vectors directly from the input
    gene embs. Because have to make sure there is no value information mixed in,
    and that is the only place to get the raw gene embs.
    2. Bare in mind to avoid short cut for the model to just predict
    value form the query. Make sure predict based on the cell_emb.
    3. Guess it will be better to use sigmoid for the query vecs.
    4. TODO: Optionally, can even try detach the gene_embs when making query_vec.
    """

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = nn.Linear(d_model, d_in)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(
        self, cell_emb: Tensor, gene_embs: Tensor
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs)) #like the Q projection in attn
            cell_emb = cell_emb.unsqueeze(2)  # (batch, embsize, 1)
            # the pred gene expr values, # (batch, seq_len)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)#genequery x W x cellemb
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            # zero logits need to based on the cell_emb, because of input exprs
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            # expand cell_emb to (batch, seq_len, embsize)
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)

class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # TODO: test using actual embedding layer if input is categorical
        # expand last dimension
        x = x.unsqueeze(-1)
        # print(x.size())
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class CategoryValueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class BatchLabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        dropout:float=0.2
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return self.dropout(x)


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len)
        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)
        # TODO: note that the return currently is only for training. Since decoder
        # is not used in the test setting for the integration task, the eval/inference
        # logic is not implemented yet. However, remember to implement it when
        # the decoder is used in any test setting. The inference logic will need
        # to sample from the bernoulli distribution with the zero_probs.


class BinExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        bin_nums: int = 51,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, bin_nums),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x)  # (batch, seq_len, bin_nums)
        return dict(pred=pred_value)


class CosineClassifier(nn.Module):
    """Cosine similarity based classifier."""
    def __init__(self, d: int, n_cls: int, scale: float = 20.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_cls, d))
        self.scale = scale
        nn.init.xavier_uniform_(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize features and weights, then compute cosine similarity
        x = F.normalize(x, dim=-1)
        W = F.normalize(self.W, dim=-1)
        return self.scale * (x @ W.t())


class PrototypeClassifier(nn.Module):
    """Prototype-based classifier (learned prototypes)."""
    def __init__(self, d: int, n_cls: int, metric: str = "cosine", scale: float = 20.0):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_cls, d))
        self.metric = metric
        self.scale = scale
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.metric == "cosine":
            x = F.normalize(x, dim=-1)
            protos = F.normalize(self.prototypes, dim=-1)
            logits = self.scale * (x @ protos.t())
        elif self.metric == "euclidean":
            dists = torch.cdist(x, self.prototypes)  # [N, C]
            logits = -dists
        else:
            raise ValueError("metric must be cosine or euclidean")
        return logits


class HybridClassifier(nn.Module):
    """Hybrid: Linear(or Cosine) + Prototype classifier."""
    def __init__(self, d: int, n_cls: int, use_cosine=True, cosine_scale=20.0, proto_metric="cosine"):
        super().__init__()
        # Linear / Cosine head
        if use_cosine:
            self.linear_head = CosineClassifier(d, n_cls, scale=cosine_scale)
        else:
            self.linear_head = nn.Linear(d, n_cls)
            nn.init.xavier_uniform_(self.linear_head.weight)
            if self.linear_head.bias is not None:
                nn.init.zeros_(self.linear_head.bias)

        # Prototype head
        self.proto_head = PrototypeClassifier(d, n_cls, metric=proto_metric, scale=cosine_scale)

        # Learnable fusion weight
        self.alpha = nn.Parameter(torch.tensor(0.5))  # α ∈ [0,1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_linear = self.linear_head(x)
        logits_proto = self.proto_head(x)
        # convex combination
        logits = self.alpha * logits_linear + (1 - self.alpha) * logits_proto
        return logits


# class ClsDecoder(nn.Module):
#     """Classification head with Pre-Norm + Residual + Dropout.
#        Supports linear / cosine / prototype / hybrid heads.
#     """
#     def __init__(
#         self,
#         d_model: int,
#         n_cls: int,
#         nlayers: int = 3,
#         activation: callable = nn.GELU,
#         dropout: float = 0.1,
#         head_type: str = "hybrid",  # "linear", "cosine", "proto", "hybrid"
#         cosine_scale: float = 20.0,
#     ):
#         super().__init__()
#         Act = activation
#
#         # Residual blocks: LayerNorm -> Linear -> Act -> Dropout
#         self.blocks = nn.ModuleList([
#             nn.ModuleDict({
#                 "norm": nn.LayerNorm(d_model),
#                 "ff": nn.Sequential(
#                     nn.Linear(d_model, d_model),
#                     Act(),
#                     nn.Dropout(dropout)
#                 )
#             })
#             for _ in range(max(nlayers - 1, 0))
#         ])
#
#         # Final normalization and dropout
#         self.head_norm = nn.LayerNorm(d_model)
#         self.head_drop = nn.Dropout(dropout)
#
#         # Choose output head
#         if head_type == "linear":
#             self.out_layer = nn.Linear(d_model, n_cls)
#             nn.init.xavier_uniform_(self.out_layer.weight)
#             if self.out_layer.bias is not None:
#                 nn.init.zeros_(self.out_layer.bias)
#         elif head_type == "cosine":
#             self.out_layer = CosineClassifier(d_model, n_cls, scale=cosine_scale)
#         elif head_type == "proto":
#             self.out_layer = PrototypeClassifier(d_model, n_cls, metric="cosine", scale=cosine_scale)
#         elif head_type == "hybrid":
#             self.out_layer = HybridClassifier(d_model, n_cls, use_cosine=True, cosine_scale=cosine_scale)
#         else:
#             raise ValueError(f"Unknown head_type {head_type}")
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Pass through residual blocks
#         for block in self.blocks:
#             x = x + block["ff"](block["norm"](x))
#         # Final norm + dropout
#         x = self.head_norm(x)
#         x = self.head_drop(x)
#         return self.out_layer(x)

class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.Dropout(p=0.2))
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class FlashTransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    The class is modified from torch.nn.TransformerEncoderLayer to support the
    FlashAttention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        device=None,
        dtype=None,
        norm_scheme="post",  # "pre" or "post"
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = FlashMHA(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=batch_first,
            attention_dropout=dropout,
            **factory_kwargs,
        )
        self.self_attn.batch_first=batch_first
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if self.norm_scheme not in ["pre", "post"]:
            raise ValueError(f"norm_scheme should be pre or post, not {norm_scheme}")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if src_mask is not None:
            raise ValueError("FlashTransformerEncoderLayer does not support src_mask")

        if not src_key_padding_mask.any().item():# return False if there isn't any True element, else return True
            # no padding tokens in src
            src_key_padding_mask_ = None
        else:
            # NOTE: the FlashMHA uses mask 0 for padding tokens, which is the opposite
            if src_key_padding_mask.dtype!=torch.bool:
                src_key_padding_mask = src_key_padding_mask.bool()
            src_key_padding_mask_ = ~src_key_padding_mask

        if self.norm_scheme == "pre":
            src = self.norm1(src)
            src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask_)[0]
            src = src + self.dropout1(src2)
            src = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
        else:
            src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask_)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src