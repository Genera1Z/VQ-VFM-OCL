import math
import time

import torch as pt
import torch.nn as nn
import torch.nn.functional as ptnf


class VQVAEZ(nn.Module):
    """"""

    def __init__(self, encode, decode, quant, alpha=0.0, retr=True):
        super().__init__()
        self.encode = encode
        self.decode = decode
        self.quant = quant
        self.register_buffer(
            "alpha", pt.tensor(alpha, dtype=pt.float), persistent=False
        )
        self.retr = retr  # return residual or not

    def forward(self, input):
        """
        input: image; shape=(b,c,h,w)
        """
        encode = self.encode(input)
        zidx, quant = self.quant(encode.permute(0, 2, 3, 1))  # bhw bhwc
        quant = quant.permute(0, 3, 1, 2)  # bchw
        residual = quant
        decode = None
        if self.decode:
            if self.alpha > 0:  # no e.detach not converge if align residual to encode
                residual = encode * self.alpha + quant * (1 - self.alpha)
            ste = __class__.naive_ste(encode, residual)
            decode = self.decode(ste)
        if self.retr:
            return encode, zidx, quant, residual, decode
        else:
            return encode, zidx, quant, decode

    @staticmethod
    def naive_ste(encode, quant):
        return encode + (quant - encode).detach()
        # Rotate STE in "Restructuring Vector Quantization with the Rotation Trick": bad.


class QuantiZ(nn.Module):

    def __init__(self, num_code, code_dim, in_dim=1024, std=0):
        super().__init__()
        self.num_code = num_code
        self.code_dim = code_dim

        # adapted from SimVQ; IBQ is hard to converge
        self.codebook = nn.Embedding(num_code, in_dim)
        for p in self.codebook.parameters():
            p.requires_grad = False
        nn.init.normal_(self.codebook.weight, mean=0, std=1)  # ==  code_dim**-0.5
        self.project = nn.Linear(in_dim, code_dim)

        # normalize and re-scale simiarities (negative distances)
        self.mu, self.sigma = __class__.chi_dist_mean_std(code_dim)
        self.register_buffer("std", pt.tensor(std, dtype=pt.float), persistent=False)

    def forward(self, input):
        """
        - input: encoded feature; shape=(b,h,w,c)
        - zidx: indexing tensor; shape=(b,h,w)
        - quant: quantized feature; shape=(b,h,w,c)
        """
        zsoft, zidx = self.match(input)
        quant = self.select(zidx)
        return zidx, quant

    def match(self, encode):
        b, h, w, c = encode.shape

        if any(_.requires_grad for _ in self.project.parameters()):
            self.__e = self.project(self.codebook.weight)  # (m,c)
        else:
            if not hasattr(self, "__e"):  # to save computation in evaluation
                self.__e = self.project(self.codebook.weight)
        e = self.__e

        z = encode.flatten(0, -2)  # (b,h,w,c)->(b*h*w,c)
        with pt.no_grad():  # cdist > cos-match
            s = -pt.cdist(z, e, p=2)  # (b*h*w,m)

        simi0 = s.unflatten(0, [b, h, w])  # (b,h,w,m)
        # ``mean`` has no effects on gumbel while ``std`` has
        simi = (simi0 - self.mu) / self.sigma  # ~N(0,1*std)

        if self.training and self.std > 0:
            # ``mean`` has no effect on gumbel while ``std`` has
            zsoft = ptnf.gumbel_softmax(simi * self.std, 1, False, dim=-1)  # (b,h,w,m)
        else:
            zsoft = simi.softmax(-1)
        zidx = zsoft.argmax(-1)  # (b,h,w)

        # ### real unchanged selection probability
        # i0 = simi.argmax(-1)
        # i1 = zidx
        # log_str = f"{self.std.item():.4f}, {(i0 == i1).float().mean().item() * 100:.4f}%"
        # print(log_str)

        return zsoft, zidx

    def select(self, zidx):
        quant = self.__e[zidx]  # (b,h,w,c)
        return quant

    @staticmethod
    def chi_dist_mean_std(c):
        """
        Euclidian distances between two Gaussian distributions follows a Chi distribution.  # TODO XXX
        """
        mean = math.sqrt(2) * math.exp(math.lgamma((c + 1) / 2) - math.lgamma(c / 2))
        std = math.sqrt((c - mean**2))
        return mean, std
