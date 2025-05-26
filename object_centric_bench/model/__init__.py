from ..utils import register_module
from .basic import (
    ModelWrap,
    Sequential,
    ModuleList,
    Embedding,
    Conv2d,
    PixelShuffle,
    ConvTranspose2d,
    Interpolate,
    Linear,
    Dropout,
    AdaptiveAvgPool2d,
    GroupNorm,
    LayerNorm,
    ReLU,
    GELU,
    SiLU,
    Mish,
    MultiheadAttention,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
    CNN,
    MLP,
    Identity,
    DINO2ViT,
    SAM2Hiera,
    EncoderVAESD,
    DecoderVAESD,
    EncoderTAESD,
    DecoderTAESD,
)
from .ocl import (
    SlotAttention,
    NormalShared,
    NormalSeparat,
    CartesianPositionalEmbedding2d,
    LearntPositionalEmbedding,
    VQVAE,
    Codebook,
    LearntPositionalEmbedding,
)
from .slatesteve import SLATE, STEVE, ARTransformerDecoder
from .dinosaur import DINOSAUR, BroadcastMLPDecoder
from .slotdiffusion import (
    SlotDiffusion,
    ConditionDiffusionDecoder,
    NoiseSchedule,
    UNet2dCondition,
)
from .vaez import VQVAEZ, QuantiZ
from .vqvfmocl import VVOTfd, VVOTfdT, VVOMlp, VVOMlpT, VVODfz, VVODfzT

[register_module(_) for _ in locals().values() if isinstance(_, type)]
