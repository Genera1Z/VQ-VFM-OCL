from pathlib import Path
import pickle as pkl
import time

import cv2
import lmdb
import numpy as np
import torch as pt
import torch.utils.data as ptud

from .utils import rgb_segment_to_index_segment, draw_segmentation_np


class PascalVOC(ptud.Dataset):
    """Visual Object Classes Challenge 2012 (VOC2012, train) + 2007 (val)
    - http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    - http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    - http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html#devkit
    - http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    """

    def __init__(
        self,
        data_file,
        extra_keys=["segment"],
        transform=lambda **_: _,
        max_spare=4,
        base_dir: Path = None,
    ):
        if base_dir:
            data_file = base_dir / data_file
        self.env = lmdb.open(
            str(data_file),
            subdir=False,
            readonly=True,
            readahead=False,
            meminit=False,
            max_spare_txns=max_spare,
            lock=False,
        )
        with self.env.begin(write=False) as txn:
            self.idxs = pkl.loads(txn.get(b"__keys__"))
        self.extra_keys = extra_keys
        self.transform = transform

    def __getitem__(self, index, compact=True):
        """
        - image: shape=(c=3,h,w), uint8
        - segment: shape=(h,w), uint8
        """
        # load sample pack
        with self.env.begin(write=False) as txn:
            sample0 = pkl.loads(txn.get(self.idxs[index]))
        sample1 = {}

        # load image and segment
        image0 = cv2.cvtColor(
            cv2.imdecode(  # cvtColor will unify images to 3 channels safely
                np.frombuffer(sample0["image"], "uint8"), cv2.IMREAD_UNCHANGED
            ),
            cv2.COLOR_BGR2RGB,
        )
        image = pt.from_numpy(image0).permute(2, 0, 1)
        sample1["image"] = image
        if "segment" in self.extra_keys:
            segment = pt.from_numpy(
                cv2.imdecode(sample0["segment"], cv2.IMREAD_GRAYSCALE)
            )
            sample1["segment"] = segment

        # conduct transformation
        sample2 = self.transform(**sample1)

        # compact segment idxs to be continuous
        if "segment" in self.extra_keys:
            if compact:
                segment = sample2["segment"]
                segment = (
                    segment.unique(return_inverse=True)[1].reshape(segment.shape).byte()
                )
                sample2["segment"] = segment

        return sample2

    def __len__(self):
        return len(self.idxs)

    @staticmethod
    def convert_dataset(
        src_dir=Path("/media/GeneralZ/Storage/Static/datasets_raw/pascalvoc/VOCdevkit"),
        dst_dir=Path("voc"),
    ):
        """
        Structure dataset as follows and run it!
        - VOC2012  # as training set
          - JPEGImages
            - *.jpg
          - SegmentationObject
            - *.png
        - VOC2007  # as validation set
          - JPEGImages
            - *.jpg
          - SegmentationObject
            - *.png
        """
        dst_dir.mkdir(parents=True, exist_ok=True)

        splits = dict(
            train=[
                "VOC2012/JPEGImages",
                "VOC2012/SegmentationObject",
            ],
            val=[
                "VOC2007/JPEGImages",
                "VOC2007/SegmentationObject",
            ],
        )

        for split, [image_dn, segment_dn] in splits.items():
            image_path = src_dir / image_dn
            segment_path = src_dir / segment_dn
            segment_files = list(segment_path.iterdir())
            segment_files.sort()

            dst_file = dst_dir / f"{split}.lmdb"
            lmdb_env = lmdb.open(
                str(dst_file),
                map_size=1024**4,
                subdir=False,
                readonly=False,
                meminit=False,
            )

            keys = []
            txn = lmdb_env.begin(write=True)
            t0 = time.time()

            for cnt, segment_file in enumerate(segment_files):
                fn, ext = segment_file.name.split(".")
                assert ext == "png"
                image_file = image_path / f"{fn}.jpg"

                with open(image_file, "rb") as f:
                    image_b = f.read()
                segment_bgr = cv2.imread(str(segment_file))  # (h,w,c=3)
                segment_rgb = cv2.cvtColor(segment_bgr, cv2.COLOR_BGR2RGB)
                segment_rgb = np.where(  # ignore borderlines
                    segment_rgb == np.array([[[224, 224, 192]]]), 0, segment_rgb
                )
                segment = rgb_segment_to_index_segment(segment_rgb)

                # image = cv2.imdecode(np.frombuffer(image_b, "uint8"), cv2.IMREAD_COLOR)
                # __class__.visualiz(image, segment, wait=0)

                sample_key = f"{cnt:06d}".encode("ascii")
                keys.append(sample_key)

                assert type(image_b) == bytes
                assert segment.ndim == 2 and segment.dtype == np.uint8

                sample_dict = dict(
                    image=image_b,  # (h,w,c=3) bytes
                    segment=cv2.imencode(".webp", segment)[1],  # (h,w) uint8
                )
                txn.put(sample_key, pkl.dumps(sample_dict))

                if (cnt + 1) % 64 == 0:  # write_freq
                    print(f"{cnt + 1:06d}")
                    txn.commit()
                    txn = lmdb_env.begin(write=True)

            txn.commit()
            txn = lmdb_env.begin(write=True)
            txn.put(b"__keys__", pkl.dumps(keys))
            txn.commit()
            lmdb_env.close()

            print(f"total={cnt + 1}, time={time.time() - t0}")

    @staticmethod
    def visualiz(image, segment=None, wait=0):
        """
        - image: bgr format, shape=(h,w,c=3), uint8
        - segment: index format, shape=(h,w), uint8
        """
        assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8

        cv2.imshow("i", image)

        if segment is not None:
            assert segment.ndim == 2 and segment.dtype == np.uint8
            segment_viz = draw_segmentation_np(image, segment, alpha=0.75)
            cv2.imshow("s", segment_viz)

        cv2.waitKey(wait)
        return image, segment_viz
