"""
1枚のWSI(.tif)から、Ultralytics YOLOがそのまま読み込める標準構成
(images/<slide_name>/*.jpg, labels/<slide_name>/*.txt, dataset.yaml)でパッチを抽出する。

パッチ候補の選び方・検出ロジックはbatch_extract_wbc_patches.pyのものをそのまま
再利用し、本スクリプトはその結果をYOLOラベル形式(クラスID 正規化cx cy w h)に
変換してimages/labels構成で書き出す部分だけを担当する。

注意:
- ここで書き出されるラベルは261021notebookの色閾値検出器による「下書き」であり、
  人手レビュー前。ディレクトリ・ファイル形式はYOLO標準に準拠しているが、中身の
  精度(見逃し・誤検出)はinspect_wbc_detection.pyでの目視チェックや
  CVAT/Label Studioでのレビューを経て別途担保すること。
- 1スライドのみのデータのため、dataset.yamlのtrain/valは暫定で同一データを指す。
  これはUltralyticsが構造を正しく読み込めるかの動作確認用であり、このまま学習に
  使うとtrain/valが同一になり精度を過大評価する。複数スライドが揃ってから、
  スライド単位でtrain/val/testに分割するスクリプトを別途用意すること。

使い方:
    python extract_yolo_patches.py \
        --tif-path /workspace/DigitalHematology/data/testdata_to_nakagawa/s260325009260810246.tif \
        --output-dir /workspace/DigitalHematology/experiments/261030/yolo_dataset_v1 \
        --mode random --n-patches 40 --negative-ratio 0.15 \
        --patch-size 640 --size 30 --blue-thresh 180 --red-thresh 130
"""

import argparse
import sys
from pathlib import Path

import yaml
from openslide import OpenSlide

# 検出ロジックはexperiments/261021/batch_extract_wbc_patches.pyのものを再利用する
sys.path.append(str(Path(__file__).resolve().parent.parent / "261021"))

from batch_extract_wbc_patches import (
    build_overview,
    find_tissue_bbox,
    _select_random,
    _select_grid,
)

CLASS_NAMES = ["wbc"]  # クラスID0 = 白血球(まずは1クラスのみ)


def box_xyxy_to_yolo_line(box, patch_size, class_id=0):
    """パッチ内ローカル座標のxyxy(ピクセル)を、YOLOラベル形式の1行に変換する。"""
    x_min, y_min, x_max, y_max = box
    cx = (x_min + x_max) / 2 / patch_size
    cy = (y_min + y_max) / 2 / patch_size
    w = (x_max - x_min) / patch_size
    h = (y_max - y_min) / patch_size
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tif-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["random", "grid"], default="random")
    parser.add_argument("--n-patches", type=int, default=40, help="mode=randomの時のみ使用")
    parser.add_argument("--negative-ratio", type=float, default=0.15)
    parser.add_argument("--patch-size", type=int, default=640)
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--blue-thresh", type=int, default=180)
    parser.add_argument("--red-thresh", type=int, default=130)
    parser.add_argument("--min-box-size", type=int, default=60)
    parser.add_argument("--downsample", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    slide_name = Path(args.tif_path).stem
    images_dir = Path(args.output_dir) / "images" / slide_name
    labels_dir = Path(args.output_dir) / "labels" / slide_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    wsi = OpenSlide(args.tif_path)
    overview = build_overview(wsi, downsample=args.downsample)
    bbox = find_tissue_bbox(overview, downsample=args.downsample)

    common_kwargs = dict(
        wsi=wsi, bbox=bbox, overview=overview, downsample=args.downsample,
        patch_size=args.patch_size, size=args.size, blue_thresh=args.blue_thresh,
        red_thresh=args.red_thresh, min_box_size=args.min_box_size, seed=args.seed,
    )
    if args.mode == "grid":
        accepted = _select_grid(negative_ratio=args.negative_ratio, **common_kwargs)
    else:
        accepted = _select_random(n_patches=args.n_patches, negative_ratio=args.negative_ratio, **common_kwargs)

    for x, y, boxes in accepted:
        patch_id = f"{slide_name}_x{x}_y{y}"
        patch_img = wsi.read_region((x, y), level=0, size=(args.patch_size, args.patch_size)).convert("RGB")
        patch_img.save(images_dir / f"{patch_id}.jpg", quality=95)

        lines = [box_xyxy_to_yolo_line(b, args.patch_size) for b in boxes]
        (labels_dir / f"{patch_id}.txt").write_text("\n".join(lines))

    wsi.close()

    yaml_path = Path(args.output_dir) / "dataset.yaml"
    yaml_path.write_text(yaml.dump({
        "path": str(Path(args.output_dir).resolve()),
        "train": f"images/{slide_name}",
        "val": f"images/{slide_name}",  # 1スライドのみの暫定値。複数スライドが揃ったら分割スクリプトで上書きする
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }, allow_unicode=True, sort_keys=False))

    print(f"完了: {len(accepted)}パッチ(images: {images_dir}, labels: {labels_dir})")
    print(f"dataset.yaml: {yaml_path}(train/valが同一データな点に注意。学習にはまだ使わないこと)")


if __name__ == "__main__":
    main()
