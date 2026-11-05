"""
検出器(pickup_wbc_thresh/detect_wbc_boxes)の見逃し(偽陰性)がないかを目視確認するための
スクリプト。1枚のWSIから指定枚数のパッチをランダムに切り出し、検出されたBBoxを緑枠で
描画して保存する。検出0件のパッチも(陽性/陰性で選り分けず)そのまま含めるため、
「本当は白血球が写っているのに枠が付いていない」パッチがあれば目視で見つけられる。

候補座標の決め方・検出ロジックはbatch_extract_wbc_patches.pyのものをそのまま再利用する。

使い方:
    python inspect_wbc_detection.py \
        --tif-path /workspace/DigitalHematology/data/testdata_to_nakagawa/s260325009260810246.tif \
        --output-dir /workspace/DigitalHematology/experiments/261030/inspection_v1 \
        --n-patches 100 \
        --patch-size 640 \
        --size 30 --blue-thresh 180 --red-thresh 130
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from openslide import OpenSlide

# 検出ロジックはexperiments/261021/batch_extract_wbc_patches.pyのものを再利用する
sys.path.append(str(Path(__file__).resolve().parent.parent / "261021"))

from batch_extract_wbc_patches import (
    build_overview,
    find_tissue_bbox,
    generate_candidate_coords,
    detect_wbc_boxes,
)


def draw_boxes(image_array, boxes, color=(0, 255, 0), thickness=2):
    """検出されたBBoxを緑枠で描画したコピーを返す(元のimage_arrayは変更しない)"""
    drawn = image_array.copy()
    for x_min, y_min, x_max, y_max in boxes:
        cv2.rectangle(drawn, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
    return drawn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tif-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-patches", type=int, default=100)
    parser.add_argument("--patch-size", type=int, default=640)
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--blue-thresh", type=int, default=180)
    parser.add_argument("--red-thresh", type=int, default=130)
    parser.add_argument("--min-box-size", type=int, default=60)
    parser.add_argument("--downsample", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wsi = OpenSlide(args.tif_path)
    slide_name = Path(args.tif_path).stem

    overview = build_overview(wsi, downsample=args.downsample)
    bbox = find_tissue_bbox(overview, downsample=args.downsample)

    coords = generate_candidate_coords(
        bbox, args.patch_size, n_candidates=args.n_patches, wsi_dims=wsi.dimensions,
        overview=overview, downsample=args.downsample, seed=args.seed,
    )
    if len(coords) < args.n_patches:
        print(f"警告: 要求{args.n_patches}枚に対し{len(coords)}枚しか座標が得られませんでした")

    n_with_detection = 0
    for x, y in coords:
        patch_img = wsi.read_region((x, y), level=0, size=(args.patch_size, args.patch_size)).convert("RGB")
        image_array = np.array(patch_img)

        boxes = detect_wbc_boxes(
            image_array, size=args.size, blue_thresh=args.blue_thresh,
            red_thresh=args.red_thresh, min_box_size=args.min_box_size,
        )
        if boxes:
            n_with_detection += 1

        drawn = draw_boxes(image_array, boxes)
        # ファイル名に検出数nを入れておくと、n0(未検出)だけを後で一覧表示して
        # 見逃しが無いか優先的にチェックできる
        patch_id = f"{slide_name}_x{x}_y{y}_n{len(boxes)}"
        cv2.imwrite(str(output_dir / f"{patch_id}.png"), cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))

    wsi.close()
    print(f"完了: {len(coords)}枚中{n_with_detection}枚で検出あり。出力先: {output_dir}")


if __name__ == "__main__":
    main()
