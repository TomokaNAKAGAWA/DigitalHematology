#!/bin/bash
# ==========================================
# [EDIT] Resource Configuration
# ==========================================
#SBATCH --job-name=2610311927_pickup-wbc-segment640_test1
#SBATCH --partition=x-large-creator
#SBATCH --output=/workspace/filesrv02/nakagawa/DigitalHematology/logs/2610311927_pickup-wbc-segment640_test1/%j_2610311927_pickup-wbc-segment640_test1.out
#SBATCH --error=/workspace/filesrv02/nakagawa/DigitalHematology/logs/2610311927_pickup-wbc-segment640_test1/%j_2610311927_pickup-wbc-segment640_test1.out
#SBATCH --signal=B:USR1@432
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=12:00:00

# リソースの根拠:
# - GPUは使わない(色閾値の検出のみ)ため --gres は指定しない。
# - 処理はPythonの1スレッドで1マスずつ進むため、CPUは2コア(本体+画像デコード/I/O)で足りる。
# - メモリの最大はオーバービュー作成時の帯1本(16001x2000px, RGBA+RGBで約0.25GB)程度。
#   パッチは640x640(約1.2MB)を1枚ずつ扱い、画像はメモリに溜めない。8gは数倍の余裕を見た値。
#   初回終了後に `sacct -j <JOBID> --format=JobID,MaxRSS,Elapsed` で実測し、見直すこと。
# - 1スライド全域(約8,850マス)の所要時間は未実測のため、時間切れを避けて長めに取る。
# ⚠️ 注意: --time を変更したら --partition(~1h:small, ~2h:medium, ~4h:large, それ以上:x-large)
#          と --signal(--timeの1%、最低30秒)も合わせて見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/nakagawa/DigitalHematology" # リポジトリの場所
export EXP_NAME="2610311927_pickup-wbc-segment640_test1" # 実験名

# =====================================================
# Storage
# 入力はSSDへコピーしない(USE_LOCAL_SSD_INPUT=1 だと data/ 全体(1枚約8GBのtif4枚)を
# 毎回rsyncしてしまうため)。出力はSSDに書き、ジョブ終了時に outputs/${EXP_NAME}/ へ回収する。
# =====================================================

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=1

# =====================================================
# python path
# =====================================================

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

# =====================================================
# Single run
# 設定はすべて config.yml で行う(experiment.py が自分の隣から読む)。
# =====================================================

RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH}" # 実行するコマンド

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
