from pathlib import Path

from object_centric_bench.datum import ClevrTex, MSCOCO, PascalVOC, MOVi


def main():
    # TODO XXX download the dataset according to ``ClevrTex`` class docs
    #   and structure the files according to  ``ClevrTex.convert_dataset`` docs
    ClevrTex.convert_dataset(
        src_dir=Path("path/to/original_clevrtex"),
        dst_dir=Path("clevrtex"),
    )
    # # TODO XXX download and structure accordingly
    # MSCOCO.convert_dataset(
    #     src_dir=Path("path/to/original_coco"),
    #     dst_dir=Path("coco"),
    # )
    # # TODO XXX download and structure accordingly
    # PascalVOC.convert_dataset(
    #     src_dir=Path("path/to/original_voc"),
    #     dst_dir=Path("coco"),
    # )
    # # TODO XXX download and structure accordingly
    # MOVi.convert_dataset(
    #     src_dir=Path("path/to/original_movi_d"),
    #     dst_dir=Path("coco"),
    # )


if __name__ == "__main__":
    main()
