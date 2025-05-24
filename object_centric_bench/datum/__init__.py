from ..utils import register_module
from .dataset import DataLoader, ChainDataset, ConcatDataset, StackDataset
from .dataset_clevrtex import ClevrTex
from .dataset_coco import MSCOCO
from .dataset_movi import MOVi
from .dataset_voc import PascalVOC
from .transform import (
    Lambda,
    Normalize,
    Concat,
    Rearrange,
    Repeat,
    Clone,
    PadTo1,
    RandomFlip,
    RandomCrop,
    CenterCrop,
    Resize,
    Slice1,
    SliceTo1,
    RandomSliceTo1,
    StridedRandomSlice1,
    Detach,
)

[register_module(_) for _ in locals().values() if isinstance(_, type)]
