max_num = 20 + 1
resolut0 = [256, 256]
resolut1 = [16, 16]
num_code = 4096
embed_dim = 256
vfm_dim = 384

total_step = 50000  # 100000 better
val_interval = total_step // 40
batch_size_t = 32 // 4  # 64 better
batch_size_v = batch_size_t
num_work = 4
lr = 2e-4 / 4  # must scale with batch_size

### datum

IMAGENET_MEAN = [[[123.675]], [[116.28]], [[103.53]]]
IMAGENET_STD = [[[58.395]], [[57.12]], [[57.375]]]
transform_t = [
    # (t=24,c,h,w) (t,n,c=4) (t,h,w)
    dict(type="StridedRandomSlice1", keys=["video", "bbox", "segment"], dim=0, size=6),
    # the following 2 == RandomResizedCrop: better than max sized random crop
    dict(
        type="RandomCrop",
        keys=["video", "segment"],
        size=None,
        scale=[0.75, 1],
        bbox_key="bbox",
    ),
    dict(type="Resize", keys=["video"], size=resolut0, interp="bilinear"),
    dict(type="Resize", keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(
        type="RandomFlip", keys=["video", "segment"], dims=[-1], bbox_key="bbox", p=0.5
    ),
    dict(type="Normalize", keys=["video"], mean=[IMAGENET_MEAN], std=[IMAGENET_STD]),
    dict(type="PadTo1", keys=["bbox"], dim=1, size=max_num, value=0),
]
transform_v = [
    dict(type="CenterCrop", keys=["video", "segment"], size=None, bbox_key="bbox"),
    dict(type="Resize", keys=["video"], size=resolut0, interp="bilinear"),
    dict(type="Resize", keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(type="Normalize", keys=["video"], mean=[IMAGENET_MEAN], std=[IMAGENET_STD]),
    dict(type="PadTo1", keys=["bbox"], dim=1, size=max_num, value=0),
]
dataset_t = dict(
    type="MOVi",
    data_file="movi_d/train.lmdb",
    extra_keys=["bbox", "segment"],
    transform=dict(type="Compose", transforms=transform_t),
    base_dir=...,
)
dataset_v = dict(
    type="MOVi",
    data_file="movi_d/val.lmdb",
    extra_keys=["bbox", "segment"],
    transform=dict(type="Compose", transforms=transform_v),
    base_dir=...,
)
collate_fn_t = None
collate_fn_v = None

### model

model = dict(
    type="STEVE",
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
    encode_posit_embed=dict(type="Identity"),
    encode_project=dict(
        type="MLP", in_dim=vfm_dim, dims=[vfm_dim * 2, embed_dim], ln="pre", dropout=0.0
    ),
    initializ=dict(type="MLP", in_dim=4, dims=[embed_dim] * 2),
    aggregat=dict(
        type="SlotAttention",
        num_iter=1,
        embed_dim=embed_dim,
        ffn_dim=embed_dim * 4,
        dropout=0.01,
        trunc_bp="bi-level",  # >None
    ),
    transit=dict(
        type="TransformerEncoderLayer",
        d_model=embed_dim,
        nhead=4,
        dim_feedforward=embed_dim * 4,
        dropout=0.1,
        activation="gelu",
        batch_first=True,
        norm_first=False,
        bias=False,
    ),
    mediat=dict(
        type="VQVAE",
        encode=dict(
            type="Sequential",
            modules=[
                dict(type="Interpolate", scale_factor=0.5, interp="bicubic"),
                dict(type="EncoderTAESD", se=[0, 14], gn=0),
                dict(type="GroupNorm", num_groups=1, num_channels=64),
                dict(
                    type="Conv2d", in_channels=64, out_channels=embed_dim, kernel_size=1
                ),
            ],
        ),
        decode=None,
        codebook=dict(type="Codebook", num_embed=num_code, embed_dim=embed_dim),
    ),
    decode=dict(
        type="ARTransformerDecoder",
        resolut=resolut1,
        embed_dim=embed_dim,
        posit_embed=dict(
            type="LearntPositionalEmbedding",
            resolut=[resolut1[0] * resolut1[1]],
            embed_dim=embed_dim,
        ),
        backbone=dict(
            type="TransformerDecoder",
            decoder_layer=dict(
                type="TransformerDecoderLayer",
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=embed_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
                bias=False,
            ),
            num_layers=4,
        ),
        readout=dict(
            type="Linear", in_features=embed_dim, out_features=num_code, bias=False
        ),
    ),
)
model_imap = dict(input="video", condit="bbox")
model_omap = ["slotz", "attent", "zidx", "recon"]
ckpt_map = [  # target<-source
    ["m.mediat.encode.", "m.encode."],
    ["m.mediat.codebook.", "m.codebook."],
]
freez = [r"m\.encode_backbone\..*", r"m\.mediat\..*"]

### learn

param_groups = None
optimiz = dict(type="Adam", params=param_groups, lr=lr)
gscale = dict(type="GradScaler")
gclip = dict(type="ClipGradNorm", max_norm=0.05)

loss_fn = dict(
    recon=dict(
        metric=dict(type="CrossEntropyLoss"),
        map=dict(input="output.recon", target="output.zidx"),
        transform=dict(  # (b,t,c,h,w)->(b*t,c,h,w) (b,t,h,w)->(b*t,h,w)
            type="Rearrange", keys=["input", "target"], pattern="b t ... -> (b t) ..."
        ),
    ),
)
_acc_dict_ = dict(
    # metric=...,
    map=dict(input="output.segment", target="batch.segment"),
    transform=dict(
        type="Rearrange", keys=["input", "target"], pattern="b t h w -> b (t h w)"
    ),
)
metric_fn_t = dict(
    mbo=dict(metric=dict(type="mBO", skip=[]), **_acc_dict_),
)
metric_fn_v = dict(
    ari=dict(metric=dict(type="ARI", skip=[]), **_acc_dict_),
    ari_fg=dict(metric=dict(type="ARI", skip=[0]), **_acc_dict_),
    mbo=dict(metric=dict(type="mBO", skip=[]), **_acc_dict_),
    miou=dict(metric=dict(type="mIoU", skip=[]), **_acc_dict_),
)

before_step = [
    dict(
        type="CbLinearCosine",
        assigns=["optimiz.param_groups[0]['lr']=value"],
        nlin=total_step // 20,
        ntotal=total_step,
        vstart=0,
        vbase=lr,
        vfinal=lr / 1e3,
    ),
]
after_forward = [
    # convert output.attent to segmentation masks: (b,t,n,h,w) -> (b,t,h,w)
    dict(type="Clone", keys=["output.attent"], keys2=["output.segment"]),
    dict(
        type="Lambda",
        keys=["output.segment"],
        func=f"lambda _: ptnf.interpolate(_.detach().flatten(0, 1), size={resolut0}, mode='bilinear').unflatten(0, [_.size(0), -1]).argmax(2).byte()",
    ),
]
callback_t = [
    dict(type="Callback", before_step=before_step, after_forward=after_forward),
    dict(type="AverageLog", log_file=...),
]
callback_v = [
    dict(type="Callback", before_step=None, after_forward=after_forward),
    callback_t[1],
    dict(type="SaveModel", save_dir=..., since_step=total_step * 0.5),
]
