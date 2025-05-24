resolut0 = [256, 256]
num_code = 4096
embed_dim0 = 4
vfm_dim = 384

total_step = 30000  # 100000 better
val_interval = total_step // 40
batch_size_t = 64
batch_size_v = batch_size_t
num_work = 4
lr = 2e-3  # >1e-3

### datum

IMAGENET_MEAN = [[[123.675]], [[116.28]], [[103.53]]]
IMAGENET_STD = [[[58.395]], [[57.12]], [[57.375]]]
transform_t = [
    # the following 2 == RandomResizedCrop: better than max sized random crop
    dict(type="RandomCrop", keys=["image"], size=None, scale=[0.75, 1]),
    dict(type="Resize", keys=["image"], size=resolut0, interp="bilinear"),
    dict(type="RandomFlip", keys=["image"], dims=[-1], p=0.5),
    dict(type="Normalize", keys=["image"], mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
transform_v = [
    dict(type="CenterCrop", keys=["image"], size=None),
    dict(type="Resize", keys=["image"], size=resolut0, interp="bilinear"),
    dict(type="Normalize", keys=["image"], mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
dataset_t = dict(
    type="ClevrTex",
    data_file="clevrtex/train.lmdb",
    extra_keys=[],
    transform=dict(type="Compose", transforms=transform_t),
    base_dir=...,
)
dataset_v = dict(
    type="ClevrTex",
    data_file="clevrtex/val.lmdb",
    extra_keys=[],
    transform=dict(type="Compose", transforms=transform_v),
    base_dir=...,
)
collate_fn_t = None
collate_fn_v = None

### model

model = dict(
    type="VVODfz",
    encode_backbone=dict(
        type="Sequential",
        modules=[
            dict(type="Interpolate", scale_factor=0.875, interp="bicubic"),
            dict(
                type="DINO2ViT",
                model_name="vit_small_patch14_dinov2.lvd142m",
                in_size=int(resolut0[0] * 0.875),
                rearrange=True,
                norm_out=True,
            ),
        ],
    ),
    encode_posit_embed=None,
    encode_project=None,
    initializ=None,
    aggregat=None,
    mediat=dict(
        type="VQVAEZ",
        encode=dict(
            type="Sequential",
            modules=[
                dict(
                    type="CNN",  # conv-norm-act-conv-...
                    in_dim=vfm_dim,
                    dims=[vfm_dim, vfm_dim, vfm_dim],
                    kernels=[3, 3, 3],
                    strides=[1, 1, 1],
                    ctypes=[0, 0, 0],
                    gn=1,
                    act="SiLU",
                ),
                dict(type="GroupNorm", num_groups=1, num_channels=vfm_dim),
                dict(
                    type="Conv2d",
                    in_channels=vfm_dim,
                    out_channels=embed_dim0,
                    kernel_size=1,
                ),
            ],
        ),
        decode=dict(
            type="Sequential",
            modules=[  # conv_shuffle+taesd(dx4) > taesd(dx8)
                dict(
                    type="Conv2d",
                    in_channels=embed_dim0,
                    out_channels=256,
                    kernel_size=1,
                ),
                dict(
                    type="CNN",
                    in_dim=256,
                    dims=[256, 64, 64],
                    kernels=[1, 1, 1],
                    strides=[1, 1, 1],
                    ctypes=[0, 2, 0],
                    gn=1,
                    act="SiLU",
                ),
                dict(type="GroupNorm", num_groups=1, num_channels=64),
                dict(type="SiLU"),
                dict(type="DecoderTAESD", se=[2, 5], gn=0),
                dict(type="DecoderTAESD", se=[6, 19], gn=0),
            ],
        ),
        quant=dict(type="QuantiZ", num_code=num_code, code_dim=embed_dim0, std=0),
        alpha=0.0,
    ),
    decode=None,
)
model_imap = dict(input="image")
model_omap = ["feature", "encode", "zidx", "quant", "residual", "decode"]
ckpt_map = None
freez = [r"m\.encode_backbone\..*"]

### learn

param_groups = None
optimiz = dict(type="Adam", params=param_groups, lr=lr)
gscale = dict(type="GradScaler")
gclip = dict(type="ClipGradNorm", max_norm=1)

loss_fn = dict(
    recon=dict(
        metric=dict(type="MSELoss"),
        map=dict(input="output.decode", target="batch.image"),
        transform=dict(
            type="Resize",
            keys=["target"],
            size=[_ // 2 for _ in resolut0],
            interp="bicubic",
        ),
    ),
    align=dict(
        metric=dict(type="MSELoss"),
        map=dict(input="output.quant", target="output.encode"),
        transform=dict(type="Detach", keys=["target"]),
    ),
    commit=dict(
        metric=dict(type="MSELoss"),  # aqcr+nosge > acq+nosge
        map=dict(input="output.encode", target="output.residual"),
        transform=dict(type="Detach", keys=["target"]),
        weight=0.25,
    ),
    norm_e=dict(
        metric=dict(type="MSELoss"),  # norm > cos
        map=dict(input="output.encode", target="output.encode"),
        transform=dict(
            type="Lambda",
            keys=["target"],
            func="lambda _: ptnf.group_norm(_.detach(), num_groups=1)",
        ),
        weight=0.1,
    ),
    lpips=dict(
        metric=dict(type="LPIPSLoss", net="alex"),
        map=dict(input="output.decode", target="batch.image"),
        transform=dict(
            type="Resize",
            keys=["target"],
            size=[_ // 2 for _ in resolut0],
            interp="bicubic",
        ),
    ),
)
metric_fn_t = metric_fn_v = dict()

before_step = [
    dict(
        type="CbLnCosine",  # simi re-scale pre gumbel softmax
        assigns=["model.m.mediat.quant.std.data[...]=value"],
        ntotal=total_step,
        vbase=1,
        vfinal=2.35,
    ),
    dict(
        type="CbCosineLinear",  # residual connection in vae
        assigns=["model.m.mediat.alpha.data[...]=value"],
        ncos=total_step // 2,  # 2 > 4 > 10
        ntotal=total_step,
        vbase=1,
        vmid=1,
        vfinal=1,
    ),
    dict(
        type="CbLinearCosine",  # learning rate
        assigns=["optimiz.param_groups[0]['lr']=value"],
        nlin=total_step // 20,
        ntotal=total_step,
        vstart=0,
        vbase=lr,
        vfinal=lr / 1e3,  # 1e3 > 1e4
    ),
]
callback_t = [
    dict(type="Callback", before_step=before_step),
    dict(type="AverageLog", log_file=...),
]
callback_v = [
    dict(type="Callback", before_step=None),
    callback_t[1],
    dict(type="SaveModel", save_dir=..., since_step=total_step * 0.5),
]
