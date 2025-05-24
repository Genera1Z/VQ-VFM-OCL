import math

from scipy.optimize import linear_sum_assignment
import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as ptnf

from .basic import MLP


class SlotAttention(nn.Module):
    """TODO XXX modularization/cgv: correct the wrong implementation!"""

    def __init__(
        self, num_iter, embed_dim, ffn_dim, dropout=0, kv_dim=None, trunc_bp=None
    ):
        """
        - dropout: only works in self.ffn; a bit is beneficial
        """
        super().__init__()
        kv_dim = kv_dim or embed_dim
        assert trunc_bp in ["bi-level", None]
        self.num_iter = num_iter
        self.trunc_bp = trunc_bp
        self.norm1q = nn.LayerNorm(embed_dim)
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm1kv = nn.LayerNorm(kv_dim)
        self.proj_k = nn.Linear(kv_dim, embed_dim, bias=False)
        self.proj_v = nn.Linear(kv_dim, embed_dim, bias=False)
        # self.dropout = nn.Dropout(dropout)  # always bad for attention
        self.rnn = nn.GRUCell(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = MLP(embed_dim, [ffn_dim, embed_dim], None, dropout)

    def forward(self, input, query, smask=None, num_iter=None):
        """
        input: in shape (b,h*w,c)
        query: in shape (b,n,c)
        smask: slots' mask, shape=(b,n), dtype=bool
        """
        b, n, c = query.shape
        self_num_iter = num_iter or self.num_iter
        kv = self.norm1kv(input)
        k = self.proj_k(kv)
        v = self.proj_v(kv)
        q = query
        for _ in range(self_num_iter):
            if _ + 1 == self_num_iter:
                if self.trunc_bp == "bi-level":  # BO-QSA
                    q = q.detach() + query - query.detach()
            x = q
            q = self.norm1q(q)
            q = self.proj_q(q)
            u, a = __class__.inverted_scaled_dot_product_attention(q, k, v, smask)
            y = self.rnn(u.flatten(0, 1), x.flatten(0, 1)).view(b, n, -1)
            z = self.norm2(y)
            q = y + self.ffn(z)  # droppath on ffn seems harmful
        return q, a

    @staticmethod
    def inverted_scaled_dot_product_attention(q, k, v, smask=None, eps=1e-5):
        scale = q.size(2) ** -0.5  # temperature
        logit = pt.einsum("bqc,bkc->bqk", q * scale, k)
        if smask is not None:
            logit = logit.where(smask[:, :, None], -pt.inf)
        a0 = logit.softmax(1)  # inverted: softmax over query  # , logit.dtype
        a = a0 / (a0.sum(2, keepdim=True) + eps)  # re-normalize over key
        # a = self_dropout(a)
        o = pt.einsum("bqv,bvc->bqc", a, v)
        return o, a0

    """@staticmethod
    def inverted_scaled_dot_product_attention(q, k, v, eps=1e-5, h=4):
        q = rearrange(q, "b q (h d) -> (b h) q d", h=h)
        k = rearrange(k, "b k (h d) -> (b h) k d", h=h)
        v = rearrange(v, "b k (h d) -> (b h) k d", h=h)
        scale = q.size(2) ** -0.5  # temperature
        logit = pt.einsum("bqc,bkc->bqk", q * scale, k)
        a0 = logit.softmax(1)  # inverted: softmax over query  # , logit.dtype
        a = a0 / (a0.sum(2, keepdim=True) + eps)  # re-normalize over key
        # a = self_dropout(a)
        o = pt.einsum("bqv,bvc->bqc", a, v)
        o = rearrange(o, "(b h) q d -> b q (h d)", h=h)
        return o, a0"""


class CartesianPositionalEmbedding2d(nn.Module):
    """"""

    def __init__(self, resolut, embed_dim):
        super().__init__()
        self.pe = nn.Parameter(
            __class__.meshgrid(resolut)[None, ...], requires_grad=False
        )
        self.project = nn.Linear(4, embed_dim)

    @staticmethod
    def meshgrid(resolut, low=-1, high=1):
        assert len(resolut) == 2
        yx = [pt.linspace(low, high, _ + 1) for _ in resolut]
        yx = [(_[:-1] + _[1:]) / 2 for _ in yx]
        grid_y, grid_x = pt.meshgrid(*yx)
        return pt.stack([grid_y, grid_x, 1 - grid_y, 1 - grid_x], 2)

    def forward(self, input):
        """
        input: in shape (b,h,w,c)
        output: in shape (b,h,w,c)
        """
        max_h, max_w = input.shape[1:3]
        output = input + self.project(self.pe[:, :max_h, :max_w, :])
        return output


class LearntPositionalEmbedding(nn.Module):
    """Support any dimension. Must be channel-last.
    PositionalEncoding: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, resolut: list, embed_dim: int):
        super().__init__()
        self.resolut = resolut
        self.embed_dim = embed_dim
        self.pe = nn.Parameter(pt.zeros(1, *resolut, embed_dim), requires_grad=True)
        nn.init.trunc_normal_(self.pe)

    def forward(self, input, retp=False):
        """
        input: in shape (b,*r,c)
        output: in shape (b,*r,c)
        """
        max_r = ", ".join([f":{_}" for _ in input.shape[1:-1]])
        # TODO XXX support variant length
        # pe = timm.layers.pos_embed.resample_abs_pos_embed(self.pe, ...)
        # pe = self.pe[:, :max_resolut, :]
        pe = eval(f"self.pe[:, {max_r}, :]")
        output = input + pe
        if retp:
            return output, pe
        return output

    def extra_repr(self):
        return f"{self.resolut}, {self.embed_dim}"


class NormalSeparat(nn.Module):
    """Separate gaussians as queries."""

    def __init__(self, num, dim):
        super().__init__()
        self.num = num
        self.dim = dim
        self.mean = nn.Parameter(pt.empty(1, num, dim))
        self.logstd = nn.Parameter(
            (pt.ones(1, num, dim) * dim**-0.5).log()
        )  # scheduled std cause nan in dinosaur; here is learnt
        nn.init.xavier_uniform_(self.mean[0, :, :])  # very important

    def forward(self, b):
        smpl = self.mean.expand(b, -1, -1)
        if self.training:
            randn = pt.randn_like(smpl)
            smpl = smpl + randn * self.logstd.exp()
        return smpl

    def extra_repr(self):
        return f"1, {self.num}, {self.dim}"


class NormalShared(nn.Module):
    """Shared gaussian as queries."""

    # TODO new trick: Conditional Random Initialization

    def __init__(self, num, dim):
        super().__init__()
        self.num = num
        self.dim = dim
        self.mean = nn.Parameter(pt.empty(1, 1, dim))
        self.logstd = nn.Parameter(pt.empty(1, 1, dim))
        nn.init.xavier_uniform_(self.mean)
        nn.init.xavier_uniform_(self.logstd)

    def forward(self, b, n=None):
        self_num = self.num
        if n is not None:
            self_num = n
        smpl = self.mean.expand(b, self_num, -1)
        randn = pt.randn_like(smpl)
        smpl = smpl + randn * self.logstd.exp()
        return smpl


class VQVAE(nn.Module):
    """
    Oord et al. Neural Discrete Representation Learning. NeurIPS 2017.

    reconstruction loss and codebook alignment/commitment (quantization) loss
    """

    def __init__(self, encode, decode, codebook):
        super().__init__()
        self.encode = encode  # should be encoder + quantconv
        self.decode = decode  # should be decoder + postquantconv
        self.codebook = codebook

    def forward(self, input):
        """
        input: image; shape=(b,c,h,w)
        """
        encode = self.encode(input)
        zsoft, zidx = self.codebook.match(encode, False)
        quant = self.codebook(zidx).permute(0, 3, 1, 2)  # (b,h,w,c) -> (b,c,h,w)
        quant2 = __class__.grad_approx(encode, quant)
        decode = None
        if self.decode:
            decode = self.decode(quant2)
        return encode, zidx, quant, decode

    @staticmethod
    def grad_approx(z, q, nu=0):  # nu=1 always harmful; maybe smaller nu???
        """
        straight-through gradient approximation

        synchronized:
        Straightening Out the Straight-Through Estimator: Overcoming Optimization Challenges in Vector Quantized Networks
        """
        assert nu >= 0
        q = z + (q - z).detach()  # delayed: default
        if nu > 0:
            q += nu * (q - q.detach())  # synchronized
        return q


class Codebook(nn.Module):
    """
    clust: always negative
    replac: always positive
    sync: always negative
    """

    def __init__(self, num_embed, embed_dim):
        super().__init__()
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.templat = nn.Embedding(num_embed, embed_dim)
        n = self.templat.weight.size(0)  # good to vqvae pretrain but bad to dvae
        self.templat.weight.data.uniform_(-1 / n, 1 / n)

    def forward(self, input):
        """
        input: indexes in shape (b,..)
        output: in shape (b,..,c)
        """
        output = self.templat(input)
        return output

    def match(self, encode, sample: bool, tau=1, detach="encode"):
        return __class__.match_encode_with_templat(
            encode, self.templat.weight, sample, tau, detach
        )

    @staticmethod
    def match_encode_with_templat(encode, templat, sample, tau=1, detach="encode"):
        """
        encode: in shape (b,c,h,w)
        templat: in shape (m,c)
        zsoft: in shape (b,m,h,w)
        zidx: in shape (b,h,w)
        """
        if detach == "encode":
            encode = encode.detach()
        elif detach == "templat":
            templat = templat.detach()
        # b, c, h, w = encode.shape
        # dist = __class__.euclidean_distance(  # (b*h*w,c) (m,c) -> (b*h*w,m)
        #    encode.permute(0, 2, 3, 1).flatten(0, -2), templat
        # )
        # dist = dist.view(b, h, w, -1).permute(0, 3, 1, 2)  # (b,m,h,w)
        # simi = -dist.square()  # better than without  # TODO XXX learnable scale ???
        dist = (  # always better than cdist.square, why ???
            encode.square().sum(1, keepdim=True)  # (b,1,h,w)
            + templat.square().sum(1)[None, :, None, None]
            - 2 * pt.einsum("bchw,mc->bmhw", encode, templat)
        )  # 1 > 0.5 > 2, 4
        simi = -dist
        if sample and tau > 0:
            zsoft = ptnf.gumbel_softmax(simi, tau, False, dim=1)
        else:
            zsoft = simi.softmax(1)
        zidx = zsoft.argmax(1)  # (b,m,h,w) -> (b,h,w)
        return zsoft, zidx
