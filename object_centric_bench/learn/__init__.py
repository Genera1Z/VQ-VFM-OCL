from ..utils import register_module
from .metric import (
    MetricWrap,
    CrossEntropyLoss,
    L1Loss,
    MSELoss,
    LPIPSLoss,
    ARI,
    mBO,
    mIoU,
)
from .optim import (
    Adam,
    AdamW,
    GradScaler,
    ClipGradNorm,
    ClipGradValue,
    NAdam,
    RAdam,
    group_params_by_keys,
)
from .callback import Callback
from .callback_log import AverageLog, SaveModel, CollectLog
from .callback_sched import (
    CbLinear,
    CbCosine,
    CbLnCosine,
    CbCosineLinear,
    CbLinearCosine,
    CbSquarewave,
)

[register_module(_) for _ in locals().values() if isinstance(_, type)]
