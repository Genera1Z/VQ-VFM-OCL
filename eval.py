from pathlib import Path
import pickle as pkl

from einops import rearrange
import cv2
import numpy as np
import torch as pt
import torch.nn.functional as ptnf

from object_centric_bench.datum import DataLoader
from object_centric_bench.datum.utils import draw_segmentation_np
from object_centric_bench.learn import MetricWrap
from object_centric_bench.model import ModelWrap
from object_centric_bench.utils import Config, build_from_config


@pt.no_grad()
def val_epoch(cfg, dataset_v, model, loss_fn, metric_fn_v, callback_v):
    cv2_resize_nearest = lambda i, x: cv2.resize(
        i, None, fx=x, fy=x, interpolation=cv2.INTER_NEAREST_EXACT
    )
    pack = Config({})
    pack.dataset_v = dataset_v
    pack.model = model
    pack.loss_fn = loss_fn
    pack.metric_fn_v = metric_fn_v
    pack.callback_v = callback_v
    pack.epoch = 0

    pack2 = Config({})
    mean = pt.from_numpy(np.array(cfg.IMAGENET_MEAN, "float32")).cuda()
    std = pt.from_numpy(np.array(cfg.IMAGENET_STD, "float32")).cuda()
    cnt = 0

    pack.model.eval()
    [_.before_epoch(**pack) for _ in pack.callback_v]

    for i, batch in enumerate(pack.dataset_v):
        pack.batch = {k: v.cuda() for k, v in batch.items()}

        [_.before_step(**pack) for _ in pack.callback_v]

        with pt.autocast("cuda", enabled=True):
            pack.output = pack.model(pack.batch)
            [_.after_forward(**pack) for _ in pack.callback_v]
            pack.loss = pack.loss_fn(**pack)
        pack.metric = pack.metric_fn_v(**pack)

        if 1:  # TODO XXX
            for image, segment in zip(pack.batch["image"], pack.output["segment2"]):
                pack.dataset_v.dataset.visualiz(
                    cv2.cvtColor(
                        (image * std + mean)
                        .clip(0, 255)
                        .byte()
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy(),
                        cv2.COLOR_BGR2RGB,
                    ),
                    segment=segment.cpu().numpy(),
                    wait=0,
                )
        # if 0:  # TODO XXX
        #     # makdir
        #     save_dn = Path(cfg.name)
        #     if not Path(save_dn).exists():
        #         save_dn.mkdir(exist_ok=True)
        #     # read gt image and segment
        #     imgs_gt = (  # image video
        #         (pack.batch["image"] * std.cuda() + mean.cuda()).clip(0, 255).byte()
        #     )
        #     segs_gt = pack.batch["segment"]
        #     # read pd attent -> pd segment
        #     segs_pd = pack.output["attent"]
        #     segs_pd = ptnf.interpolate(segs_pd, cfg.resolut0, mode="bilinear").argmax(1)
        #     # segs_pd = (
        #     #     ptnf.interpolate(segs_pd.flatten(0, 1), cfg.resolut0, mode="bilinear")
        #     #     .argmax(1)
        #     #     .unflatten(0, [imgs_gt.size(0), -1])
        #     # )
        #     # visualize gt image,
        #     t = 12
        #     for img_gt, seg_gt, seg_pd in zip(imgs_gt, segs_gt, segs_pd):
        #         # img_gt, seg_gt, seg_pd = [
        #         #     _[t] for _ in (img_gt, seg_gt, seg_pd)  # for t-th frame of a video
        #         # ]
        #         img_gt = cv2.cvtColor(
        #             img_gt.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR
        #         )
        #         seg_gt = seg_gt.cpu().numpy()
        #         seg_pd = seg_pd.cpu().numpy()
        #         save_path = save_dn / f"{cnt:06d}"
        #         cv2.imwrite(f"{save_path}-i.png", img_gt)
        #         cv2.imwrite(f"{save_path}-s.png", draw_segmentation_np(img_gt, seg_gt))
        #         cv2.imwrite(f"{save_path}-p.png", draw_segmentation_np(img_gt, seg_pd))
        #         # cv2.waitKey(0)
        #         # cv2.destroyAllWindows()
        #         cnt += 1
        #         break  # TODO XXX
        #     # gt segment, pd segment

        [_.after_step(**pack) for _ in pack.callback_v]

    [_.after_epoch(**pack) for _ in pack.callback_v]

    for cb in pack.callback_v:
        if cb.__class__.__name__ == "AverageLog":
            pack2.log_info = cb.mean()
            break
    return pack2


def main(  # TODO XXX
    cfg_file="config-vqdino/vqdino_mlp_r-coco-r384.py",
    ckpt_file="/media/GeneralZ/Storage/Active/20250213/New Folder/r384/archive-vqdino-42/vqdino_mlp_r-coco-r384/best.pth",
):
    data_dir = "/media/GeneralZ/Storage/Static/datasets"  # TODO XXX
    pt.backends.cudnn.benchmark = True

    cfg_file = Path(cfg_file)
    data_path = Path(data_dir)
    ckpt_file = Path(ckpt_file)

    assert cfg_file.name.endswith(".py")
    assert cfg_file.is_file()
    cfg_name = cfg_file.name.split(".")[0]
    cfg = Config.fromfile(cfg_file)
    cfg.name = cfg_name

    ## datum init

    cfg.dataset_t.base_dir = cfg.dataset_v.base_dir = data_path

    dataset_v = build_from_config(cfg.dataset_v)
    dataload_v = DataLoader(
        dataset_v,
        cfg.batch_size_v // 2,  # TODO XXX
        shuffle=False,
        num_workers=cfg.num_work,
        pin_memory=True,
    )

    ## model init

    model = build_from_config(cfg.model)
    # print(model)
    model = ModelWrap(model, cfg.model_imap, cfg.model_omap)

    if ckpt_file:
        model.load(ckpt_file, None, verbose=False)
    if cfg.freez:
        model.freez(cfg.freez, verbose=False)

    model = model.cuda()
    # model.compile()

    ## learn init

    loss_fn = MetricWrap(**build_from_config(cfg.loss_fn))
    metric_fn_v = MetricWrap(detach=True, **build_from_config(cfg.metric_fn_v))

    cfg.callback_v = [_ for _ in cfg.callback_v if _.type != "SaveModel"]
    for cb in cfg.callback_v:
        if cb.type == "AverageLog":
            cb.log_file = None
    callback_v = build_from_config(cfg.callback_v)

    ## do eval

    with pt.inference_mode(True):
        pack2 = val_epoch(cfg, dataload_v, model, loss_fn, metric_fn_v, callback_v)

    return pack2.log_info


if __name__ == "__main__":
    main()
