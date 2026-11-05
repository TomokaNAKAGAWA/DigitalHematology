"""
WSI(.tif)の組織領域全体を640×640のグリッドで読み取り、色閾値の検出器
(261021ノートブック由来のdetect_wbc_boxes)で白血球を検出して、
陽性/陰性の比率を指定したうえでYOLO形式のデータセットとして書き出す。

出力(run_dir = outputs/<EXP_NAME>/<run_name>/):
    images/<slide_name>/<slide_name>_x{X}_y{Y}.png   枠なしの元画像
    labels/<slide_name>/<slide_name>_x{X}_y{Y}.txt   "0 cx cy w h"(正規化座標)。陰性は空ファイル
    previews/<slide_name>/*.png                      labelsの.txtから枠を描き戻した確認用画像
    dataset.yaml                                     Ultralytics用の設定
    summary.json                                     スライドごとの枚数などの集計

注意:
- ラベルは色閾値検出器の出力そのもの(人手レビュー前)。
- dataset.yamlのtrain/valは同じスライド群を指す暫定値。学習前にスライド単位で分割すること。
- input_pathにディレクトリを指定すると、直下の*.tifを全て処理する。

実行: run_slurm.sh経由(PROJECT_ROOT/EXP_NAMEが必要)。設定はconfig.ymlで行う。
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from openslide import OpenSlide

CLASS_NAMES = ["wbc"]  # クラスID0 = 白血球

# ==== 関数定義(ここから) ====


def build_overview(wsi, downsample=16, strip_height=2000):
    """WSI全体を高さ方向に細切れで読み込み、間引きながら結合した低解像度画像を作る"""
    w, h = wsi.dimensions
    ov_w = max(1, w // downsample)
    rows = []
    for y in range(0, h, strip_height):
        sh = min(strip_height, h - y)
        strip = wsi.read_region((0, y), level=0, size=(w, sh)).convert("RGB")
        strip = strip.resize((ov_w, max(1, sh // downsample)))
        rows.append(np.array(strip))
    return np.vstack(rows)


def find_tissue_bbox(overview, downsample, white_thresh=5):
    """オーバービューから背景(白)でない最大の連結領域を探し、level0座標のbboxを返す"""
    gray = cv2.cvtColor(overview, cv2.COLOR_RGB2GRAY)
    tissue_mask = (gray < (255 - white_thresh)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(tissue_mask)
    if n <= 1:
        raise ValueError("組織領域が検出できませんでした。white_threshを調整してください。")
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, bw, bh, _ = stats[largest]
    return x * downsample, y * downsample, bw * downsample, bh * downsample


def generate_grid_coords(bbox, patch_size, wsi_dims, overview, downsample,
                         min_coverage=0.5, white_thresh=5):
    """bbox内をpatch_size刻みの格子状(重複なし)に走査し、組織の被覆率がmin_coverage以上の
    座標(左上, level0)を全て返す。"""
    tx, ty, tw, th = bbox
    w, h = wsi_dims

    gray_ov = cv2.cvtColor(overview, cv2.COLOR_RGB2GRAY)
    tissue_mask_ov = gray_ov < (255 - white_thresh)

    def coverage(x, y):
        ox0, oy0 = int(x / downsample), int(y / downsample)
        ox1 = max(ox0 + 1, int((x + patch_size) / downsample))
        oy1 = max(oy0 + 1, int((y + patch_size) / downsample))
        region = tissue_mask_ov[oy0:oy1, ox0:ox1]
        return region.mean() if region.size else 0.0

    x_hi = min(tx + tw, w) - patch_size
    y_hi = min(ty + th, h) - patch_size

    coords = []
    for y in range(int(ty), int(y_hi) + 1, patch_size):
        for x in range(int(tx), int(x_hi) + 1, patch_size):
            if coverage(x, y) >= min_coverage:
                coords.append((x, y))
    return coords


def detect_wbc_boxes(image_array, size=30, blue_thresh=180, red_thresh=130, step=10, min_box_size=60):
    """画像配列(H, W, 3)を受け取り、白血球候補のBBox [(x_min, y_min, x_max, y_max), ...] を返す。"""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    img_pro = cv2.bitwise_and(image_array, image_array, mask=binary)

    h, w = image_array.shape[:2]
    hits = []
    for row in range(0, h, step):
        for col in range(0, w, step):
            patch = img_pro[row:row + size, col:col + size]
            if patch.size == 0:
                continue
            blue_mean = patch[:, :, 2].mean()
            red_mean = patch[:, :, 0].mean()
            if blue_mean > blue_thresh and red_mean < red_thresh:
                hits.append((row, col))

    # 近接するヒットを1つの矩形にまとめる
    regions = []  # [row_min, row_max, col_min, col_max]
    merge_margin = 2 * size
    for row, col in hits:
        merged = False
        for idx, (rmin, rmax, cmin, cmax) in enumerate(regions):
            if (rmin - merge_margin <= row <= rmax + merge_margin) and \
               (cmin - merge_margin <= col <= cmax + merge_margin):
                regions[idx] = [min(rmin, row), max(rmax, row + size),
                                min(cmin, col), max(cmax, col + size)]
                merged = True
                break
        if not merged:
            regions.append([row, row + size, col, col + size])

    boxes = []
    for rmin, rmax, cmin, cmax in regions:
        if (rmax - rmin) > min_box_size and (cmax - cmin) > min_box_size:
            x_min = max(0, cmin - size / 2)
            x_max = min(w, cmax + size / 2)
            y_min = max(0, rmin - size / 2)
            y_max = min(h, rmax + size / 2)
            boxes.append((x_min, y_min, x_max, y_max))
    return boxes


def read_patch(wsi, x, y, patch_size):
    """level0の(x, y)を左上とするpatch_size四方の領域をRGBのPIL画像で返す"""
    return wsi.read_region((x, y), level=0, size=(patch_size, patch_size)).convert("RGB")


def box_xyxy_to_yolo_line(box, patch_size, class_id=0):
    """パッチ内ローカル座標のxyxy(ピクセル)を、YOLOラベル形式の1行に変換する"""
    x_min, y_min, x_max, y_max = box
    cx = (x_min + x_max) / 2 / patch_size
    cy = (y_min + y_max) / 2 / patch_size
    w = (x_max - x_min) / patch_size
    h = (y_max - y_min) / patch_size
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def export_slide(tif_path, output_dir, patch_size=640, negative_ratio=0.15, image_ext="png",
                 detector_params=None, min_coverage=0.5, downsample=16, seed=0,
                 max_candidates=None, log=print):
    """1枚のWSIをグリッドで全走査し、陽性は全件、陰性はnegative_ratioに合わせて間引いて
    output_dir/images/<slide>/ と output_dir/labels/<slide>/ に書き出す。集計dictを返す。

    negative_ratio: 書き出すパッチ全体に占める陰性(検出0件)の割合。0 <= negative_ratio < 1。
    max_candidates: 動作確認用。指定するとグリッドのマスからこの数だけ無作為に選んで処理する。
                    Noneなら全マスを処理する。
    """
    if not 0 <= negative_ratio < 1:
        raise ValueError(f"negative_ratioは0以上1未満で指定してください: {negative_ratio}")
    if image_ext not in ("png", "jpg"):
        raise ValueError(f"image_extは'png'か'jpg'を指定してください: {image_ext}")
    if detector_params is None:
        raise ValueError("detector_paramsを指定してください")

    tif_path = Path(tif_path)
    if not tif_path.is_file():
        raise FileNotFoundError(f"TIFFが見つかりません: {tif_path}")

    t0 = time.time()
    slide_name = tif_path.stem
    images_dir = Path(output_dir) / "images" / slide_name
    labels_dir = Path(output_dir) / "labels" / slide_name
    # 前回の結果と混ざらないよう、既に書き出し済みなら止める
    for d in (images_dir, labels_dir):
        if d.exists() and any(d.iterdir()):
            raise FileExistsError(f"{d} は空ではありません。別のOUTPUT_DIR(run_name)を指定するか、削除してください。")
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    wsi = OpenSlide(str(tif_path))
    try:
        log(f"[{slide_name}] サイズ{wsi.dimensions}。オーバービュー作成中...")
        overview = build_overview(wsi, downsample=downsample)
        bbox = find_tissue_bbox(overview, downsample=downsample)
        candidates = generate_grid_coords(bbox, patch_size, wsi.dimensions, overview, downsample,
                                          min_coverage=min_coverage)
        n_grid = len(candidates)
        rng = np.random.default_rng(seed)
        if max_candidates is not None and max_candidates < n_grid:
            idx = np.sort(rng.choice(n_grid, size=max_candidates, replace=False))
            candidates = [candidates[i] for i in idx]
        log(f"[{slide_name}] 組織bbox={bbox}, グリッド{n_grid}マス中{len(candidates)}マスを検出にかけます")

        # 1周目: 全マスを検出にかけ、陽性/陰性に振り分ける(画像はメモリに溜めない)
        positives, negatives = [], []
        for i, (x, y) in enumerate(candidates):
            boxes = detect_wbc_boxes(np.array(read_patch(wsi, x, y, patch_size)), **detector_params)
            (positives if boxes else negatives).append((x, y, boxes))
            if (i + 1) % 500 == 0:
                log(f"[{slide_name}]   {i + 1}/{len(candidates)}マス処理済み(陽性{len(positives)})")

        if not positives:
            raise RuntimeError(
                f"[{slide_name}] 陽性パッチが0件でした。検出パラメータや入力画像を確認してください。"
            )

        n_negative_target = round(len(positives) * negative_ratio / (1 - negative_ratio))
        if n_negative_target > len(negatives):
            log(f"[{slide_name}] 警告: 陰性が不足しています(必要{n_negative_target}, 実際{len(negatives)})。"
                "陰性は全件を使います")
            n_negative_target = len(negatives)
        idx = rng.choice(len(negatives), size=n_negative_target, replace=False)
        kept_negatives = [negatives[i] for i in sorted(idx)]

        # 2周目: 採用したマスだけを読み直して保存する
        for x, y, boxes in positives + kept_negatives:
            patch_id = f"{slide_name}_x{x}_y{y}"
            img = read_patch(wsi, x, y, patch_size)
            if image_ext == "jpg":
                img.save(images_dir / f"{patch_id}.jpg", quality=95)
            else:
                img.save(images_dir / f"{patch_id}.png")
            lines = [box_xyxy_to_yolo_line(b, patch_size) for b in boxes]
            (labels_dir / f"{patch_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    finally:
        wsi.close()

    stats = {
        "slide": slide_name,
        "tif_path": str(tif_path),
        "tissue_bbox_xywh": [int(v) for v in bbox],
        "n_grid": n_grid,
        "n_processed": len(candidates),
        "n_positive": len(positives),
        "n_negative_found": len(negatives),
        "n_negative_kept": len(kept_negatives),
        "n_boxes": sum(len(b) for _, _, b in positives),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    log(f"[{slide_name}] 完了: 陽性{stats['n_positive']}枚(枠{stats['n_boxes']}個), "
        f"陰性{stats['n_negative_kept']}/{stats['n_negative_found']}枚, {stats['elapsed_sec']}秒")
    return stats


def write_dataset_yaml(output_dir, slide_names, dataset_root=None):
    """Ultralytics用のdataset.yamlを書く。train/valは暫定で同じスライド群を指す。
    dataset_root: yamlの`path`に書く場所。Noneならoutput_dir。書き出し後に別の場所へ
    移す場合(scratchからの回収など)は移動先を指定する。"""
    image_dirs = [f"images/{s}" for s in slide_names]
    header = (
        "# 暫定: train/valが同じデータを指している。学習前にスライド単位で分割すること。\n"
    )
    body = yaml.safe_dump({
        "path": str(Path(dataset_root or output_dir).resolve()),
        "train": image_dirs,
        "val": image_dirs,
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }, allow_unicode=True, sort_keys=False)
    yaml_path = Path(output_dir) / "dataset.yaml"
    yaml_path.write_text(header + body)
    return yaml_path


def draw_label_previews(output_dir, slide_name, n_previews=20, seed=0):
    """書き出した.txtラベルを読み戻して枠を描いた確認用画像を previews/<slide>/ に保存する。
    陽性(枠あり)を優先して最大n_previews枚を無作為に選ぶ。保存したパスのリストを返す。"""
    images_dir = Path(output_dir) / "images" / slide_name
    labels_dir = Path(output_dir) / "labels" / slide_name
    preview_dir = Path(output_dir) / "previews" / slide_name
    preview_dir.mkdir(parents=True, exist_ok=True)

    label_paths = sorted(labels_dir.glob("*.txt"))
    positive = [p for p in label_paths if p.read_text().strip()]
    negative = [p for p in label_paths if not p.read_text().strip()]
    rng = np.random.default_rng(seed)
    n_pos = min(n_previews, len(positive))
    n_neg = min(n_previews - n_pos, len(negative))
    chosen = [positive[i] for i in rng.choice(len(positive), size=n_pos, replace=False)] + \
             [negative[i] for i in rng.choice(len(negative), size=n_neg, replace=False)]

    saved = []
    for label_path in chosen:
        image_paths = list(images_dir.glob(f"{label_path.stem}.*"))
        if len(image_paths) != 1:
            raise FileNotFoundError(f"{label_path.name}に対応する画像が1つに定まりません: {image_paths}")
        img = cv2.imread(str(image_paths[0]))
        h, w = img.shape[:2]
        for line in label_path.read_text().strip().splitlines():
            _, cx, cy, bw, bh = map(float, line.split())
            x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x1, y1 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        out_path = preview_dir / f"{label_path.stem}.png"
        cv2.imwrite(str(out_path), img)
        saved.append(out_path)
    return saved


# ==== 関数定義(ここまで) ====


def resolve_tif_paths(input_path):
    """input_pathが.tifファイルならそれ1つ、ディレクトリなら直下の*.tifを全て返す"""
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() not in (".tif", ".tiff"):
            raise ValueError(f".tifファイルではありません: {input_path}")
        return [input_path]
    if input_path.is_dir():
        tif_paths = sorted(input_path.glob("*.tif"))
        if not tif_paths:
            raise FileNotFoundError(f"{input_path} に.tifファイルがありません")
        return tif_paths
    raise FileNotFoundError(f"input_pathが存在しません: {input_path}")


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def setup_logger(run_dir: Path, name: str = "experiment") -> logging.Logger:
    """コンソールとrun_dir/experiment.logの両方に出力するloggerを作る"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(run_dir / "experiment.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    config = yaml.safe_load((Path(__file__).parent / "config.yml").read_text())
    detector_params = config["detector"]
    input_path = Path(config["input_path"])
    if not input_path.is_absolute():
        input_path = project_root / input_path
    tif_paths = resolve_tif_paths(input_path)
   
    # run_nameが出力先ディレクトリ名になる。同じrun_nameで完了済みなら即終了する
    run_dir = get_run_dir(project_root, __file__, config["run_name"], output_root=output_root)
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=config["run_name"],
                       input_path=str(input_path), n_slides=len(tif_paths))

    logger.info(f"run_dir: {run_dir}")
    logger.info(f"対象スライド{len(tif_paths)}枚: {[p.name for p in tif_paths]}")

    all_stats = []
    for i, tif_path in enumerate(tif_paths):
        logger.info(f"[{i + 1}/{len(tif_paths)}] {tif_path.name}")
        stats = export_slide(
            tif_path, run_dir,
            patch_size=config["patch_size"],
            negative_ratio=config["negative_ratio"],
            image_ext=config["image_ext"],
            detector_params=detector_params,
            min_coverage=config["min_coverage"],
            downsample=config["downsample"],
            seed=config["seed"] + i,  # スライドごとに異なるが再現可能なシード
            max_candidates=config["max_candidates"],
            log=logger.info,
        )
        draw_label_previews(run_dir, stats["slide"], n_previews=config["n_previews"], seed=config["seed"])
        all_stats.append(stats)
        # 途中で落ちても処理済みスライドの集計が残るよう、毎回書き直す
        (run_dir / "summary.json").write_text(
            json.dumps({"config": config, "slides": all_stats}, indent=2, ensure_ascii=False)
        )

    # OUTPUT_ROOT(scratch)に書いた場合もジョブ終了時にoutputs/へ回収されるので、回収先を書く
    canonical_dir = project_root / "outputs" / exp_name / config["run_name"]
    yaml_path = write_dataset_yaml(run_dir, [s["slide"] for s in all_stats], dataset_root=canonical_dir)
    logger.info(f"dataset.yaml: {yaml_path}")

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
