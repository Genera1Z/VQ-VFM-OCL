from einops import rearrange, repeat
import torch as pt
import torch.nn as nn


class DINOSAUR(nn.Module):

    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        aggregat,
        decode,
    ):
        super().__init__()
        self.encode_backbone = encode_backbone
        self.encode_posit_embed = encode_posit_embed
        self.encode_project = encode_project
        self.initializ = initializ
        self.aggregat = aggregat
        self.decode = decode
        self.reset_parameters(
            [self.encode_posit_embed, self.encode_project, self.aggregat]
        )

    @staticmethod
    def reset_parameters(modules):
        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.GRUCell):
                    if m.bias:
                        nn.init.zeros_(m.bias_ih)
                        nn.init.zeros_(m.bias_hh)

    def forward(self, input, condit=None):
        """
        - input: image, shape=(b,c,h,w)
        - condit: condition, shape=(b,n,c)
        """
        feature = self.encode_backbone(input).detach()  # (b,c,h,w)
        b, c, h, w = feature.shape
        encode = feature.permute(0, 2, 3, 1)  # (b,h,w,c)
        encode = self.encode_posit_embed(encode)
        encode = encode.flatten(1, 2)  # (b,h*w,c)
        encode = self.encode_project(encode)

        query = self.initializ(b if condit is None else condit)  # (b,n,c)
        slotz, attent = self.aggregat(encode, query)
        attent = rearrange(attent, "b n (h w) -> b n h w", h=h)

        clue = [h, w]
        recon, attent2 = self.decode(clue, slotz)  # (b,h*w,c)
        recon = rearrange(recon, "b (h w) c -> b c h w", h=h)
        attent2 = rearrange(attent2, "b n (h w) -> b n h w", h=h)

        return feature, slotz, attent, attent2, recon
        # segment acc: attent < attent2


class BroadcastMLPDecoder(nn.Module):  # TODO BroadcastCNNDecoder
    """DINOSAUR's decoder."""

    def __init__(self, posit_embed, backbone):
        super().__init__()
        self.posit_embed = posit_embed
        self.backbone = backbone

    def forward(self, input, slotz, smask=None):
        """
        - input: destructed target, shape=(b,m,c)
        - slotz: slots, shape=(b,n,c)
        - smask: slots' mask, shape=(b,n), dtype=bool
        """
        h, w = input
        b, n, c = slotz.shape

        mixture = repeat(slotz, "b n c -> (b n) hw c", hw=h * w)
        mixture = self.posit_embed(mixture)
        mixture = self.backbone(mixture)

        recon, alpha = mixture[:, :, :-1], mixture[:, :, -1:]
        recon = rearrange(recon, "(b n) hw c -> b n hw c", b=b)
        alpha = rearrange(alpha, "(b n) hw 1 -> b n hw 1", b=b)
        if smask is not None:
            alpha = alpha.where(smask[:, :, None, None], -pt.inf)
        # faster than pt.einsum()
        alpha = alpha.softmax(1)
        recon = (recon * alpha).sum(1)  # (b,hw,c)

        attent2 = alpha[:, :, :, 0]  # (b,n,hw)
        return recon, attent2
