# VQ-VFM-OCL (VVO) -- Vector Quantized Vision Foundation Models for Object Centric Learning



## About

Official implementation of paper "**Vector-Quantized Vision Foundation Models for Object-Centric Learning**" available on [arXiv:2502.20263](https://arxiv.org/abs/2502.20263).

Supported OCL methods include, categorized by OCL decoding:
- Auto-regressive decoding: [SLATE](https://github.com/singhgautam/slate) vs VVO-Tfd, [STEVE](https://github.com/singhgautam/steve) vs VVO-TfdT, [SPOT](https://github.com/gkakogeorgiou/spot) vs VVO-Tfd9
- Mixture-based decoding: [DINOSAUR](https://github.com/martius-lab/videosaur) vs VVO-Mlp, [VideSAUR](https://github.com/martius-lab/videosaur) vs VVO-SmdT
- Diffusion-based decoding: [SlotDiffusion](https://github.com/Wuziyi616/SlotDiffusion) vs VVO-Dfz

Object discovery performance with DINO2 ViT (s/14) for OCL encoding. VVO is instantiated as VQDINO; Tfd, TfdT, Mlp and Dfz are Transformer, Transformer-temporal, MLP and Diffusion for OCL decoding respectively.

<img src="res/acc_vqdino_all.png" style="width:100%;">

Using higher resolution.

<img src="res/acc_vqdino_r384_coco.png" style="width:50%;">

Qualitative  results.

<img src="res/qualitative.png" style="width:100%;">



## Stucture

```
- config-slatesteve  # configs for SLATE and STEVE
    └ *.py
- config-dinosaur  # configs for DINOSAUR
    └ *.py
- config-slotdiffusion  # configs for SlotDiffusion
    └ *.py
- config-vqdino  # configs forr VQDINO (VVO with DINO for OCL encoding)
    └ *.py
- object_centric_bench
    └ datum  # implementations of datasets ClevrTex, COCO, VOC and MOVi
        └ *.py
    └ model  # modules that compose OCL models
        └ *.py
    └ learn  # metrics, callbacks and logging
        └ *.py
    └ *.py
- convert.py
- train.py
- eval.py
- requirements.txt
```



## Features

- **fp16 fast training** Auto-mixed precision training (fp16) is enabled. Most of the training can be finished in 10 hours using one V100 GPU.
- **less I/O overhead** Datasets are stored in [LMBD](https://lmdb.readthedocs.io) database format to save I/O overhead, beneficial especially on computing cluster.
- **config-driven experiment** This is totally config-driven framework, largely inspired by [OpenMMLab](https://github.com/open-mmlab), but with much less capsulation.



## TODO

- SPOT & VVO-Tfd9: To be integrated into this framework;
- VideoSAUR & VVO-SmdT: To be integrated into this framework.



## How to Use

- Install requirements: ```pip install -r requirements.txt```. Use package versions no older than the specification.
- Convert original datasets into LMDB format: ```python convert.py```. But firstly download original datasets according to docs of ```XxxDataset.convert_dataset()```.
- Run train and eval: ```python train.py``` and ```python eval.py```. But firstly change the arguments marked with ```TODO XXX``` to your needs.



## Converted Datasets

Converted datasets, including ClevrTex, COCO, VOC and MOVi-D are available here.



## Model Checkpoints

All checkpoints for the models in the two tables above are available here.



## Tips

1. Any config file can be converted into typical Python code by changing from
```Python
...
model = dict(type="class_name", key1=value1,..)
...
```
to
```Python
from object_centric_bench.datum import *
from object_centric_bench.model import *
from object_centric_bench.learn import *
...
model = class_name(key1=value1,..)
...
```

2. All config files follow a similar structure, and you can use file comparator [Meld](https://meldmerge.org) with VSCode plugin [Meld Diff](https://marketplace.visualstudio.com/items?itemName=danielroedl.meld-diff) to check their differences.



## About

I am now working on object-centric learning. If you have any ideas about this please do not hesitate to contact me.
- WeChat: Genera1Z
- email: rongzhen.zhao@aalto.fi, zhaorongzhenagi@gmail.com
