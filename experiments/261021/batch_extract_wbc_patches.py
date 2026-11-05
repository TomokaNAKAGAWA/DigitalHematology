"""
261021_pickup-wbc_Re-nakagawa.ipynb の検出ロジック(色味+サイズによる白血球候補検出)を
フォルダ内の複数WSIに対してバッチ適用し、YOLO学習データの「下書き」(画像+矩形候補)を作るスクリプト。

設計上のポイント(セッション内の議論を反映):
- 検出器が「白血球あり」と判定したパッチだけでなく、「検出0」のパッチも一定割合
  (negative_ratio)だけ混ぜて集める。理由:
    1. 検出器の見逃し(偽陰性)パターンの白血球を学習データから完全に排除してしまうのを防ぐ
    2. YOLOの学習には「物体が無い」ことを示す背景のみの画像も必要
- ここで出力するBBoxはあくまで「下書き」。人手でのレビュー(CVAT/Label Studio等での
  追加・削除・位置修正)を経てから、最終的にYOLO形式(クラスID+正規化xywh)に変換する想定。
  このスクリプト単体でYOLOの学習データが完成するわけではない。
- パッチサイズは既定で640(YOLOの入力解像度)。最初から等倍で切り出すことで、
  後からリサイズして解像度(=見た目のスケール)が学習時と推論時でズレる事態を避ける。
- 同一スライド(個体)由来のパッチをmanifestに記録し、後でtrain/val/splitを
  「パッチ単位」ではなく「スライド単位」で行えるようにする。

使い方:
    # mode=random(既定): 乱択でpatches-per-slide枚を集める。動作確認・小規模な試行向け。
    python batch_extract_wbc_patches.py \
        --input-dir /workspace/DigitalHematology/data/testdata_to_nakagawa \
        --output-dir /workspace/DigitalHematology/experiments/261021/wbc_dataset_v1 \
        --mode random \
        --patches-per-slide 40 \
        --negative-ratio 0.15 \
        --size 30 --blue-thresh 180 --red-thresh 130 \
        --patch-size 640 --seed 0

    # mode=grid: 組織全体を隙間なく敷き詰めて網羅的に集める。本番のデータセット作成向け。
    # patches-per-slideは無視され、枚数は組織の面積(=検出された陽性数)で決まる。
    python batch_extract_wbc_patches.py \
        --input-dir /workspace/DigitalHematology/data/testdata_to_nakagawa \
        --output-dir /workspace/DigitalHematology/experiments/261021/wbc_dataset_v1 \
        --mode grid \
        --negative-ratio 0.15 \
        --size 30 --blue-thresh 180 --red-thresh 130 \
        --patch-size 640 --seed 0

注意:
- 対象のtifファイルは1枚あたり約8GBあり、build_overview()はWSI全体を間引き読みするため
  1スライドあたり数分かかる可能性がある。まずは --mode random --patches-per-slide を
  小さくし、1スライドだけ(input-dirに1ファイルだけコピー/シンボリックリンクするか、下の
  mainを書き換えて対象ファイルを絞る)で動作確認してから --mode grid での全件処理に
  進むことを推奨する。
- --mode gridは組織全体を1枚ずつ検出にかけるため、--mode randomよりかなり時間がかかる。
  また生成されるパッチ(特に陽性)の枚数が多くなるため、CVAT/Label Studioでの人手レビュー
  工数も相応に増える点に注意。
- detect_wbc_boxesのsize, blue_thresh, red_threshは、これまでの対話で有効性が確認された
  size=30, blue_thresh=180, red_thresh=130を既定値にしているが、他のスライドでも
  同様に機能するかは別途確認が必要。
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from openslide import OpenSlide


# ---------------------------------------------------------------------------
# 1. 組織領域の検出とパッチ候補座標のサンプリング
#    (261021ノートブック cell-6 の build_overview / find_tissue_bbox を再利用)
# ---------------------------------------------------------------------------

# WSI全体の縮小プレビューを作る
def build_overview(wsi, downsample=16, strip_height=2000):
    """WSI全体を高さ方向に細切れで読み込み、間引きながら結合した低解像度画像を作る"""
    w, h = wsi.dimensions # wsi.dimensionが返す(幅, 高さ)のような2要素のタプルを1行でwとhに代入
    ov_w = max(1, w // downsample)
    ## //...割り算結果から小数点以下を切り捨てた整数を返す
    rows = [] # 結果を貯めていく時の手法
    ## rows = []で空リストを作り、rows.append(...)でループのたびに要素を追加していく
    for y in range(0, h, strip_height): # 0からh未満までstrip_height刻みでyを動かすループ
    ## range(start, stop, step)の形
        sh = min(strip_height, h - y)
        strip = wsi.read_region((0, y), level=0, size=(w, sh)).convert("RGB") # 
        ## (0, y)を左上として、level0（最大解像度）から幅w・高さshの領域を読む(ライブラリ固有の使い方)
        ## .convert("RGB")のようにメソッドを.でつなげて書くのはメソッドチェーン。読み込む→RGB変換するを1行で続けて書いている
        strip = strip.resize((ov_w, max(1, sh // downsample)))
        rows.append(np.array(strip))
        ## np.array(strip)...Pillowの画像オブジェクトをNumPy配列に変換
    return np.vstack(rows) # リストの中の複数のNumpy配列を縦(Vertical)方向に結合する関数。
    ## 短冊状に読んだrowsを1枚の画像につなぎ合わせている


# プレビューから組織の範囲を見つける
def find_tissue_bbox(overview, downsample, white_thresh=5):
    """オーバービューから背景(白)でない最大の連結領域を探し、level0座標のbboxを返す"""
    gray = cv2.cvtColor(overview, cv2.COLOR_RGB2GRAY)
    # OpenCVでカラー画像をグレースケールに変換
    tissue_mask = (gray < (255 - white_thresh)).astype(np.uint8)
    # .astype(np.uint8)...True/Falseを0/1の整数に変換(OpenCVの関数が整数配列を要求するため)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(tissue_mask)
    # 連結成分分析...マスク画像の中で隣接する白(=組織)ピクセルの塊ごとにグループ分けし，各グループの面積・位置を返す
    ## タプルのアンパックで4つの戻り値を受け取っている
    ## _...この値は使わないので名前を付けないという慣習的な変数名
    if n <= 1:
        raise ValueError("組織領域が検出できませんでした。white_threshを調整してください。")
    # 組織が1つも見つからなかった場合のErrorを示すコード

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    # np.argmax(stats[1:, cv2.CC_STAT_AREA])... 面積が最大のインデックスを返す
    ## stats[1:, ...]…スライス。先頭(0番目=背景)を除いた1番目以降の行全てを意味
    ## argmaxの結果に1を足すのは，スライス絵0番目を飛ばした分のインデックスのズレを補正するため
    x, y, bw, bh, _ = stats[largest]
    # その最大領域の「x, y, 幅, 高さ, 面積」のうち面積(_)以外を取り出している
    return x * downsample, y * downsample, bw * downsample, bh * downsample
    # x * downsampleで縮小率を掛け戻しているのは，縮小したプレビュー画像上の計算を元のlevel0の座標系に変換するため

# パッチ候補座標をランダムに作る
def generate_candidate_coords(bbox, patch_size, n_candidates, wsi_dims, overview, downsample,
                               min_coverage=0.5, white_thresh=5, margin_ratio=0.05, seed=0):
    """組織の被覆率がmin_coverage以上の候補座標をn_candidates件作る(重複除去なし)。

    n_candidatesは最終的に欲しいパッチ数より多めに指定し、呼び出し側で
    検出結果(陽性/陰性)を見ながら必要数まで採用していく想定。
    """
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

    margin_x, margin_y = tw * margin_ratio, th * margin_ratio
    x_lo, x_hi = tx + margin_x, min(tx + tw, w) - patch_size - margin_x
    y_lo, y_hi = ty + margin_y, min(ty + th, h) - patch_size - margin_y
    if x_hi < x_lo:
        x_lo, x_hi = tx, min(tx + tw, w) - patch_size
    if y_hi < y_lo:
        y_lo, y_hi = ty, min(ty + th, h) - patch_size

    rng = np.random.default_rng(seed)
    coords = []
    max_attempts = n_candidates * 300
    for _ in range(max_attempts):
        if len(coords) >= n_candidates:
            break
        x = rng.uniform(x_lo, x_hi)
        y = rng.uniform(y_lo, y_hi)
        if coverage(x, y) >= min_coverage:
            coords.append((int(x), int(y)))
    return coords


# 組織のbboxをpatch_size刻みで隙間なく(重複なし)敷き詰めて、候補座標を網羅的に列挙する
def generate_grid_coords(bbox, patch_size, wsi_dims, overview, downsample,
                          min_coverage=0.5, white_thresh=5):
    """bbox内をpatch_size刻みの格子状(重複なし)に走査し、組織の被覆率がmin_coverage以上の
    座標を全て返す。generate_candidate_coordsと違い乱数は使わず、件数は組織の面積で決まる。
    """
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


# ---------------------------------------------------------------------------
# 2. 白血球候補の検出
#    (261021ノートブック cell-7 の pickup_wbc_thresh から、可視化部分を除き
#     BBoxをそのまま返すように整理。スキャン時のx/y軸の扱いは、元コードの
#     PIL.crop実装だと正方形パッチ限定で座標ラベルが転置されたまま使われて
#     いたため、YOLO用に正しい(x, y)座標で出力できるよう素直なnumpyスライスに
#     書き直した。正方形パッチである限り走査される領域の集合は元コードと同一。)
# ---------------------------------------------------------------------------

def detect_wbc_boxes(image_array, size=30, blue_thresh=180, red_thresh=130, step=10, min_box_size=60):
    """画像配列(H, W, 3)を受け取り、白血球候補のBBox [(x_min, y_min, x_max, y_max), ...] を返す。

    min_box_size: マージ後の矩形の縦横がこの値を超えなければノイズとみなして捨てる、
    最終的な緑枠の最小サイズ(px)。以前はsize/2から自動算出していたが、
    sizeを変えても最小サイズフィルタだけは独立して固定できるよう切り出した。
    """
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

    # 近接するヒットを1つの矩形にまとめる(元のarea_lstマージロジックを踏襲)
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


# ---------------------------------------------------------------------------
# 3. 1スライド分の処理(候補選定 + 検出 + 保存)
# ---------------------------------------------------------------------------

def _select_random(wsi, bbox, overview, downsample, patch_size, n_patches, negative_ratio,
                    size, blue_thresh, red_thresh, min_box_size, seed):
    """乱択モード: 必要数より多めの候補座標を作り、検出数に応じて陽性/陰性を
    指定枚数(n_patches, negative_ratio)まで振り分けながら採用する。
    戻り値は [(x, y, boxes), ...] のリスト(画像の保存はprocess_slide側で行う)。
    """
    candidates = generate_candidate_coords(
        bbox, patch_size, n_candidates=n_patches * 4, wsi_dims=wsi.dimensions,
        overview=overview, downsample=downsample, seed=seed,
    )

    n_negative_target = max(1, round(n_patches * negative_ratio))
    n_positive_target = n_patches - n_negative_target

    accepted = []
    pos_count, neg_count = 0, 0

    for (x, y) in candidates:
        if pos_count >= n_positive_target and neg_count >= n_negative_target:
            break

        image_array = np.array(wsi.read_region((x, y), level=0, size=(patch_size, patch_size)).convert("RGB"))
        boxes = detect_wbc_boxes(image_array, size=size, blue_thresh=blue_thresh, red_thresh=red_thresh,
                                  min_box_size=min_box_size)

        is_positive = len(boxes) > 0
        if is_positive and pos_count >= n_positive_target:
            continue
        if not is_positive and neg_count >= n_negative_target:
            continue

        accepted.append((x, y, boxes))
        if is_positive:
            pos_count += 1
        else:
            neg_count += 1

    return accepted


def _select_grid(wsi, bbox, overview, downsample, patch_size, negative_ratio,
                  size, blue_thresh, red_thresh, min_box_size, seed):
    """網羅モード: 組織のbbox全体を隙間なく敷き詰めて全候補を検出にかけ、
    陽性は全件採用、陰性はnegative_ratio通りの比率になるよう間引いて採用する。
    枚数は組織の面積で決まり、n_patchesのような上限指定は無い。
    """
    candidates = generate_grid_coords(
        bbox, patch_size, wsi_dims=wsi.dimensions, overview=overview, downsample=downsample,
    )

    positives, negatives = [], []
    for (x, y) in candidates:
        image_array = np.array(wsi.read_region((x, y), level=0, size=(patch_size, patch_size)).convert("RGB"))
        boxes = detect_wbc_boxes(image_array, size=size, blue_thresh=blue_thresh, red_thresh=red_thresh,
                                  min_box_size=min_box_size)
        (positives if boxes else negatives).append((x, y, boxes))

    if positives:
        n_negative_target = max(1, round(len(positives) * negative_ratio / (1 - negative_ratio)))
    else:
        # 陽性が1件も無い場合は間引かず全陰性を残す(検出パラメータ不具合に気付けるようにするため)
        n_negative_target = len(negatives)

    if len(negatives) > n_negative_target:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(negatives), size=n_negative_target, replace=False)
        negatives = [negatives[i] for i in idx]

    return positives + negatives


def process_slide(tif_path, output_dir, patch_size=640, n_patches=40, negative_ratio=0.15,
                   size=30, blue_thresh=180, red_thresh=130, min_box_size=60,
                   downsample=16, seed=0, mode="random"):
    slide_name = Path(tif_path).stem
    slide_out = Path(output_dir) / slide_name
    slide_out.mkdir(parents=True, exist_ok=True)

    wsi = OpenSlide(str(tif_path))
    mpp_x = wsi.properties.get("openslide.mpp-x")

    overview = build_overview(wsi, downsample=downsample)
    bbox = find_tissue_bbox(overview, downsample=downsample)

    if mode == "grid":
        accepted = _select_grid(wsi, bbox, overview, downsample, patch_size, negative_ratio,
                                 size, blue_thresh, red_thresh, min_box_size, seed)
    else:
        accepted = _select_random(wsi, bbox, overview, downsample, patch_size, n_patches,
                                   negative_ratio, size, blue_thresh, red_thresh, min_box_size, seed)

    manifest_rows = []
    for (x, y, boxes) in accepted:
        patch_img = wsi.read_region((x, y), level=0, size=(patch_size, patch_size)).convert("RGB")

        patch_id = f"{slide_name}_x{x}_y{y}"
        img_path = slide_out / f"{patch_id}.png"
        patch_img.save(img_path)

        draft_path = slide_out / f"{patch_id}_draft.json"
        with open(draft_path, "w") as f:
            json.dump({
                "slide": slide_name,
                "slide_x": x,
                "slide_y": y,
                "patch_size": patch_size,
                "detector_params": {"size": size, "blue_thresh": blue_thresh, "red_thresh": red_thresh},
                "boxes_xyxy": boxes,
            }, f, indent=2)

        manifest_rows.append({
            "patch_id": patch_id,
            "slide": slide_name,
            "slide_x": x,
            "slide_y": y,
            "patch_size": patch_size,
            "n_detections": len(boxes),
            "mpp_x": mpp_x,
            "image_path": str(img_path),
            "draft_label_path": str(draft_path),
        })

    wsi.close()
    return manifest_rows


# ---------------------------------------------------------------------------
# 4. フォルダ内の全スライドをループするエントリーポイント
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="WSI(.tif)が入ったフォルダ")
    parser.add_argument("--output-dir", required=True, help="抽出結果の出力先フォルダ")
    parser.add_argument("--mode", choices=["random", "grid"], default="random",
                         help="random: 乱択でpatches-per-slide枚を集める(既定、動作確認向け)。"
                              "grid: 組織全体を隙間なく敷き詰めて網羅的に集める"
                              "(patches-per-slideは無視され、枚数は組織の面積で決まる)")
    parser.add_argument("--patches-per-slide", type=int, default=40,
                         help="mode=randomの時のみ使用")
    parser.add_argument("--negative-ratio", type=float, default=0.15,
                         help="検出0件のパッチを混ぜる割合")
    parser.add_argument("--size", type=int, default=30) # size指定
    parser.add_argument("--blue-thresh", type=int, default=180)
    parser.add_argument("--red-thresh", type=int, default=130)
    parser.add_argument("--min-box-size", type=int, default=60,
                         help="検出された矩形がこのサイズ(px)を超えなければノイズとして捨てる、最終的な緑枠の最小サイズ")
    parser.add_argument("--patch-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_paths = sorted(input_dir.glob("*.tif"))
    if not tif_paths:
        raise SystemExit(f"{input_dir} に.tifファイルが見つかりません。")

    all_rows = []
    for i, tif_path in enumerate(tif_paths):
        print(f"[{i + 1}/{len(tif_paths)}] {tif_path.name} を処理中...")
        try:
            rows = process_slide(
                tif_path, output_dir,
                patch_size=args.patch_size,
                n_patches=args.patches_per_slide,
                negative_ratio=args.negative_ratio,
                size=args.size,
                blue_thresh=args.blue_thresh,
                red_thresh=args.red_thresh,
                min_box_size=args.min_box_size,
                seed=args.seed + i,  # スライドごとに異なるが再現可能なシード
                mode=args.mode,
            )
            all_rows.extend(rows)
            print(f"  -> {len(rows)}パッチ抽出(陽性/陰性の内訳はmanifestのn_detections列で確認)")
        except Exception as e:
            print(f"  !! {tif_path.name} の処理に失敗: {e}")
            continue

    manifest_path = output_dir / "manifest.csv"
    if all_rows:
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    print(f"完了: 合計{len(all_rows)}パッチ。manifest: {manifest_path}")


if __name__ == "__main__":
    main()
