# DigitalHematology
Tools for computational hematology

# Experiment Management System

SlurmおよびPBS (qsub/Miyabi) ベースの実験管理システムです。自動でスケジューラを判別し、実験の作成・投入・通知・再開をコマンド一つで行えます。

## 💡 基本コンセプト

本システムは、主に以下の2点によって動作します。

1. **シェル環境の拡張**: 便利な実験管理コマンド（`runx`, `cdx`, `lsx`, `cancelx` など）を自身のシェル環境に読み込み、実験の移動や実行を簡単に行えるようにします。
2. **実験テンプレートの自動適用 (`templates/` のコピー)**: `make create_exp` コマンドを使用して、事前に定義された実験テンプレート（コード、設定ファイル、ジョブスクリプト）を実験ディレクトリごとに展開し、実験の独立した開発・追跡を可能にします。

---

## はじめに: この環境の仕組み（Slurm が初めての方へ）

計算機クラスタは **ログインノード**（`ssh` でログインした先。軽い作業用）と
**計算ノード**（GPU や大量の CPU を積んだ実行用マシン）に分かれています。
重い処理はログインノードで直接実行せず、**ジョブ**としてスケジューラ（Slurm）に
預け、順番が来たら計算ノードで実行してもらう、という流れになります。

| 用語 | 意味 |
|---|---|
| ジョブ | スケジューラに実行を依頼する処理のかたまり。投入すると `job_id`（数字）が振られる |
| パーティション | 計算ノードのグループ。「どのマシン群で最長どれだけ動かせるか」を表す。投入時に必ず指定する |
| `sbatch` | ジョブを投入する Slurm のコマンド。本テンプレートでは `runx` が代行する |
| `squeue` | 自分のジョブの現在の状態（待機中 / 実行中）を見るコマンド |
| `scancel` | ジョブを止めるコマンド。本テンプレートでは `cancelx` が代行する |

本テンプレートは、この一連の流れ（ジョブスクリプトの用意 → 投入 → ログ回収 →
成否の記録 → Slack 通知）をまとめて自動化したものです。

---

## セットアップ

所要時間の目安は、イメージのビルドと依存のインストールを含めて **1〜2時間**です
（大半は待ち時間で、手を動かすのは数分です）。以下の 0〜4 を順番に実行してください（5 は任意）。

### 0. 事前に確認すること

**(a) 自分が使えるパーティション名**

以降のコマンドはすべて `p=<partition>` を要求します。使える名前は次で確認できます。

```bash
sinfo -o "%P %l %D" | column -t    # パーティション名 / 最大実行時間 / ノード数
```

**(b) リポジトリの置き場所**

> [!IMPORTANT]
> `make create_exp` は、リポジトリの絶対パスからパーティションの所有者を推定します。
> 現在対応しているのは以下のパスの配下だけです。
>
> `/workspace/andre01` `/workspace/david01` `/workspace/david02`
> `/workspace/filesrv01` `/workspace/filesrv02` `/workspace/grace01` `/workspace/grace02`
>
> **これ以外の場所に clone すると `make create_exp` が
> `❌ Could not infer partition owner from path:` で停止し、実験を作成できません。**
> 別のパスやクラスタで使う場合は、`tools/create_exp.sh` の `_resolve_partition()` に
> 自分の環境の対応行を追記してください。

### 1. 初期セットアップ

```bash
make setup
```

`.env` と `logs/` `notebooks/` `data/` `outputs/` `libraries/` `experiments/` を作成します（冪等。何度実行しても安全です）。

✅ **確認**: `✅ Initial setup completed` と表示され、`ls` で上記ディレクトリが見えれば成功です。

### 2. Apptainerイメージ（SIF）のビルド

本テンプレートは**すべてのジョブを Apptainer コンテナ内で実行します**。Apptainer は
「OS ごと固めた1個のファイル（`.sif`）の中でプログラムを動かす」仕組みで、誰の環境でも
同じライブラリ構成で実験を再現するために使います。最初に一度、計算ノード上で
イメージをビルドしてください。

```bash
make build_env p=<partition>
```

> [!NOTE]
> これは**ジョブとして投入される**ため、コマンド自体は `Submitted batch job 12345` と
> 表示してすぐ戻ってきます。ビルドが終わったわけではありません。

```bash
squeue -u $USER                      # 実行中かどうかを見る（消えたら終了）
tail -f logs/build_env_<job_id>.out  # 進捗ログを追う
```

✅ **確認**: ログに `Build successful!` が出て、`ls -lh env/env.sif` でファイル（数GB）が
見えれば成功です。所要時間は 20〜60 分程度です。

`env/env.def`（CUDA 12.8.1 + cuDNN / Ubuntu 24.04 + uv）から `env/env.sif` を生成します。`*.sif` は `.gitignore` 済みなのでコミットされません。

- ビルドは1回だけ。`pyproject.toml` の依存を変えただけなら再ビルドは不要で、`make uv_sync` だけで済みます（イメージに入っているのは OS パッケージと `uv` 本体のみ）。
- ビルド失敗時に既存の動くイメージを壊さないよう、`env/env.sif.new` に作ってから差し替えます。
- 一時領域は `/scratch/$USER`（ノード付属SSD）を優先して使います。CUDA devel ベースのイメージは展開に数十GBを要するため、`/tmp` では溢れることがあります。

**既にビルド済みのイメージがある場合**は、このステップを飛ばして `.env` にパスを書いてください。

```bash
# .env
SIF_PATH=/path/to/your/env.sif
```

`SIF_PATH` の解決順序は `export` 済みの値 → `.env` → `env/env.sif`（既定）です（`scripts/sif_env.sh`）。`make uv_sync` / `make jupyter` / ジョブ本体（`scripts/slurm_entry.sh`）はすべてこの同じ規則を使います。

### 3. Python環境の同期

```bash
# ベース環境のみ（分野を問わず使うものだけ）
make uv_sync p=<partition>

# 分野特化の依存を追加する
make uv_sync p=<partition> extras="llm chem"
make uv_sync p=<partition> extras=all
```

コンテナ内で `uv sync` を実行し、`.venv` を作ります。イメージが見つからない場合は**その場で失敗し**、`make build_env` の案内を出します。

これも**ジョブとして投入される**ので、すぐには終わりません。ログは
`logs/uv-sync-log_<job_id>.out` です。

✅ **確認**: ログに `uv sync completed successfully.` が出て、
`ls .venv/bin/python` が見つかれば成功です。所要時間は 5〜20 分程度です。

extra はあとから足せます。必要になった時点で `make uv_sync p=<partition> extras="llm"`
を実行し直してください（`.venv` が更新されます）。

依存は `pyproject.toml` で**ベースと分野特化に分けてあります**。ベースには分野を問わず使うもの（numpy/pandas/polars/scikit-learn/matplotlib/torch/transformers 等）だけを置き、特化したものは extra として切り出しています。

| extra | 内容 | 用途 |
|---|---|---|
| `llm` | `vllm` `trl` `peft` `bitsandbytes` `sentencepiece` `evaluate` | LLMの推論・軽量学習 |
| `chem` | `rdkit` | ケモインフォマティクス |
| `nlp` | `spacy` `scispacy` `fast-langdetect` `lxml` `elasticsearch` | 文献テキストマイニング |

インストール規模の目安（linux/cp312/x86_64、ダウンロードサイズ）:

| 構成 | パッケージ数 | サイズ |
|---|---:|---:|
| ベースのみ | 175 | 3.76 GB |
| `extras="chem"` | 176 | 3.80 GB |
| `extras="chem nlp"` | 209 | 3.85 GB |
| `extras="llm"` | 293 | 4.91 GB |
| `extras=all` | 323 | 4.99 GB |

サイズの約75%は `torch` と NVIDIA の CUDA wheel なので、`torch` を使う限りここは削れません。extra を絞る主な利点はパッケージ数・依存衝突源・同期時間の削減です。

> [!NOTE]
> `make build_env` / `make uv_sync` / `make jupyter` は現在 Slurm 環境のみのサポートとなっています。PBS (qsub) 環境については、今後のアップデートをお待ちいただくか、手動で実行してください。

### 4. シェルヘルパーの設定

ジョブ管理用コマンド（`runx`, `cdx`, `lsx`, `cancelx` など）を自身のシェル環境で使えるようにするため、`~/.bashrc` から本レポジトリの `shell/template-daily.sh` を source します。テンプレート側は関数定義だけで、Git のグローバル設定変更や `jq` 未導入時のシェル終了は行いません。

リポジトリのルートで一度だけ実行してください。すでに同じ設定があれば追記しません。

```bash
repo_root="$(git rev-parse --show-toplevel)"
loader_line="source \"${repo_root}/shell/template-daily.sh\""
grep -Fqx "${loader_line}" ~/.bashrc || printf '\n%s\n' "${loader_line}" >> ~/.bashrc
source ~/.bashrc
```

✅ **確認**: `type runx` と打って `runx is a function` と返れば成功です。
返らない場合は `source ~/.bashrc` をやり直すか、新しいターミナルを開いてください。

> [!NOTE]
> このローダーが読み込むのは `.bashrc.d/0-bootstrap.sh` と `.bashrc.d/1-experiments.sh`
> だけです。日次ブランチ運用のコマンド（`gstart` / `gpush`）は
> `.bashrc.d/2-git.sh` にあり、Git のグローバル設定を書き換えるため
> **意図的に読み込んでいません**。使いたい場合は自分の `~/.bashrc` から
> 明示的に source してください（運用の詳細は [`USAGE.md`](./USAGE.md) 参照）。

### 5. Slack通知の設定（任意）

`~/.bashrc` に以下を設定します。

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

---

## ディレクトリ構成

```
.
├── experiments/          # 実験ごとの設定・コード
│   └── 0001_20260101_baseline/
│       ├── run_slurm.sh  # リソース設定・実行コマンド
│       ├── experiment.py
│       └── config.yml
├── outputs/              # Pythonスクリプトの出力先
│   └── 0001_20260101_baseline/
├── logs/                 # Slurmログ・メタデータ
│   └── 0001_20260101_baseline/
│       └── 1234/         # job_id
│           ├── slurm.out
│           ├── run_metadata.yaml
│           ├── bootstrap_failure.log  # 初期化失敗時のみ
│           └── command.sh
├── env/                  # Apptainerイメージの定義
│   ├── env.def           # ビルド定義（コミット対象）
│   └── env.sif           # make build_env の生成物（.gitignore済み）
├── lib/                  # 共通ライブラリコード
├── templates/            # 実験作成時のテンプレートファイル
├── shell/                # ~/.bashrc から source するシェルヘルパーのloader
├── examples/             # 実験コードのお手本カタログ
├── scripts/              # ジョブ実行・監視用の内部スクリプト
│   ├── slurm_entry.sh
│   ├── sif_env.sh        # SIF_PATH の解決規則（3者で共有）
│   ├── preflight_check.py
│   └── notify_slack.sh
└── tools/                # 各種管理ツールの実体スクリプト
    ├── build_env.sh
    ├── create_exp.sh
    ├── resume_exp.sh
    ├── rename_exp.sh
    ├── mark_failed.sh
    ├── cancel_job.sh
    ├── uv_sync.sh
    ├── start_jupyter.sh
    ├── review.sh
    └── first_setup.sh
```

---

## 主な機能と対応スケジューラ

**投入（`runx`）は Slurm (`sbatch`) 専用です。** PBS (qsub/Miyabi) 環境では `runx` は使えません。
一方、ジョブ実行中の状態管理（`slurm_entry.sh`）とキャンセル（`cancelx`）は Slurm/PBS
両方に対応しています（PBS環境で既に投入済みのジョブを管理する用途を想定）。

- **パーティション自動判定**: `make create_exp` 実行時に、リポジトリのパスから所有者を、
  テンプレートの `#SBATCH --time` の初期値から時間スケール（`small` / `medium` / `large` /
  `x-large`）を決め、`small-{owner}` 等を `run_slurm.sh` に焼き込みます。
  **作成時に一度きりの計算**なので、あとから `--time` を変えても `--partition` と
  `--signal` は追従しません（変えたら手動で直してください）。テンプレートの初期値は
  `--time=196:00:00` なので、既定では必ず `x-large-{owner}` になります。
- **依存ジョブ指定**: `run_slurm.sh` 内の `#SBATCH --dependency=afterok:<job_id>` を
  有効化することで指定します。job_idは依存先実験の `outputs/{exp}/latest_job_id.txt` を参照します。
- **アレイジョブ実行**: `run_slurm.sh` 内でarray/seqモードを有効化することで、探索パラメータに
  応じた複数ジョブの一括投入をSlurmのアレイジョブ機能（`--array`）で実行します。
- **ジョブキャンセル**: `cancelx <job_id>` で自動的に `scancel` または `qdel` を呼び出してジョブを停止します（Slurm/PBS両対応）。
- **終了理由の記録**: 初期化失敗は `bootstrap_failure.log` と `run_metadata.yaml` の `fail_reason` に、実行後の終了状態は `run_metadata.yaml` に記録します。通知失敗は本体ジョブを停止させず、標準エラーに警告として残します。
- **ログ自動管理**: ジョブ終了時に標準出力・標準エラーを `logs/{exp_name}/{job_id}/` 以下に `slurm.out` / `pbs.err` として自動回収します。
- **Slack通知機能**: ジョブの開始・終了・失敗（タイムアウトやOOMを含む）を判定し、Slackへ通知します（アレイジョブにも対応）。Webhook・`jq`・通信の異常は警告のみです。

---

## 実験のライフサイクル

### 1. 実験を作成する

```bash
make create_exp name=fetch_patients
```

4桁の連番と日付が自動で付き、テンプレートから3つのファイルがコピーされます。

```
experiments/0001_20260911_fetch_patients/
├── run_slurm.sh    # リソース指定と実行コマンド（= sbatch に渡すスクリプト本体）
├── experiment.py   # 実験のロジック。ここを書く
├── config.yml      # 固定ハイパーパラメータ
└── metadata.yaml   # 自動生成（作成日時・git commit）
```

名前は `{動詞}_{対象}` のスネークケース（例: `fetch_patients` / `train_mamba` /
`eval_baseline`）を推奨します。`test_xxx` は pytest と紛らわしいため避けてください。

### 2. 3つのファイルを編集する

役割がはっきり分かれています。**迷ったらこの表に従ってください。**

| ファイル | ここに書くもの | 書かないもの |
|---|---|---|
| `config.yml` | 探索しない固定値（学習率、エポック数など） | 振りたいパラメータ |
| `run_slurm.sh` | 必要な GPU/CPU/メモリ/時間、振りたいパラメータ | 実験ロジック |
| `experiment.py` | 実験ロジック本体 | ハイパーパラメータの直書き |

**リソースの指定（`run_slurm.sh` の先頭）**

```bash
#SBATCH --gres=gpu:1          # GPU の枚数（不要なら gpu:0）
#SBATCH --cpus-per-task=20    # CPU コア数
#SBATCH --mem=110g            # メモリ
#SBATCH --time=196:00:00      # 最大実行時間。超えると強制終了される
```

> [!WARNING]
> 初期値の `--time=196:00:00` は 8 日間で、かなり長めです。**要求が大きいほど順番待ちは
> 長くなる**ので、実際に必要な時間まで縮めるのを推奨します。ただし `--time` を変えても
> `--partition` と `--signal` は自動では変わりません。時間スケール
> （60分以下=`small` / 120分以下=`medium` / 240分以下=`large` / それ以上=`x-large`）を
> またぐ変更をしたら、`#SBATCH --partition=` も手で直してください。

**実行コマンド（`run_slurm.sh` の中ほど）**

```bash
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH}"
```

`config.yml` は `experiment.py` が自分の隣から自動で読むため、ここで渡す必要はありません。
ここに足してよいのは、`experiment.py` の `parse_args()` で `add_argument()` 済みの引数だけです
（未定義の引数を渡すと Python が起動直後にエラーで終了します）。

複数の組み合わせをまとめて投入したい場合は、同じファイル内の Array run / Seq run
セクションを有効化します（詳細は [`USAGE.md`](./USAGE.md) 参照）。

### 3. 投入前にチェックする（推奨）

```bash
make preflight
```

最新の実験について、`config.yml` と `experiment.py` のキー不整合、GRID の組み合わせ数、
`#SBATCH --array=0-N` の N と組み合わせ数の一致、`outputs/` へのハードコード書き込みを
静的に検査します。`✅ preflight OK` が出れば投入してかまいません。

> [!NOTE]
> `runx` は `make preflight` を自動では呼びません。投入前に自分で実行してください。

### 4. ジョブを投入する

```bash
runx            # 最新の実験を投入
runx 1          # 実験ID を指定して投入
```

`Submitted batch job 12345` と出れば受理されています。`12345` が `job_id` です。

> [!IMPORTANT]
> `runx` は**未コミットの変更があると投入を拒否します**。どのコードで出た結果かを
> 後から追えるようにするためです。先にコミットしてください。
> どうしてもそのまま投入したい場合のみ `runx --allow-dirty` を使えます。
>
> なお、このチェックが見ているのは**変更済みファイルだけ**で、未追跡（untracked）の
> ファイルは対象外です。`make create_exp` で作ったばかりの実験ディレクトリは
> まるごと未追跡なので、チェックを素通りします。記録される `git_commit` が実際に
> 動いたコードを含まない状態になるため、**投入前に実験ディレクトリを
> `git add` してコミットしておくのが確実です。**

投入後は次のように追えます。

```bash
squeue -u $USER                                  # 待機中/実行中の一覧（消えたら終了）
tail -f logs/<exp_name>/<job_id>_<exp_name>.out  # 実行中のログを追う
cat    logs/<exp_name>/latest/run_metadata.yaml  # 今の status を見る
```

> [!NOTE]
> **実行中とジョブ終了後で、標準出力の置き場所が変わります。**
> 実行中は Slurm が `logs/{exp_name}/{job_id}_{exp_name}.out` に直接書いており、
> ジョブ終了時にこれが `logs/{exp_name}/{job_id}/slurm.out` へ移動されます。
> 実行中に `logs/{exp_name}/latest/slurm.out` を見ても、まだ存在しません。

依存関係を指定したい場合（実験1が終わったら実験2を投入、等）は、実験2の
`run_slurm.sh` にある `#SBATCH --dependency=afterok:<job_id>` を有効化し、
実験1の `outputs/{exp}/latest_job_id.txt` の値を書き込んでから `runx` します。

### 5. 実験を確認・移動する

```bash
lsx       # 実験一覧を表示
cdx       # 最新の実験ディレクトリへ移動
cdx 1     # ID指定で移動
groot     # リポジトリのルートへ戻る
```

### 6. ジョブをキャンセルする

```bash
cancelx <job_id>
cancelx <job_id> <reason>      # 理由を run_metadata.yaml に残す
```

> [!CAUTION]
> `cancelx --all` は **`squeue` に出ている自分のジョブを全部止めます**。
> このリポジトリと無関係なジョブや `make jupyter` で立てたサーバも巻き込むので、
> 普段は `job_id` を指定してください。

### 実験ごとに独自のルール・スキルを持たせたい場合

`experiments/<id>_.../CLAUDE.md` や `experiments/<id>_.../.claude/skills/` を置くと、
その実験を触っている間だけ有効な追加ルール・専用スキルとして機能します
（Claude Codeがサブディレクトリ単位で自動的にlazy-loadする標準機能で、agent設定側の変更は不要）。
詳細は [`TEMPLATE_CONCEPT.md`](./TEMPLATE_CONCEPT.md#4-実験ごとに独自のルール・スキルを持たせるオプトイン) を参照。

---

## 実験の管理

### リネーム

```bash
make rename_exp name=<exp_id_or_name> new=<new_name>
```

`experiments/` `outputs/` `logs/` の3つをまとめて改名し、`latest` シンボリックリンクも
張り直します。ID と作成日は変わりません。

### 失敗マーク

```bash
make mark_fail name=<exp_id_or_name> reason=<reason>
```

ディレクトリ名に `_FAILED_<date>_<reason>` が付与されます（`experiments/` `outputs/`
`logs/` の3つとも）。失敗の経緯は `FAILED_REASON.txt` に残ります。

> [!NOTE]
> ID は変わりません。そのため、失敗マークした実験が最大IDのままだと、引数なしの
> `runx` がそれを再投入してしまいます。マーク後は `runx <id>` のように
> **ID を明示して投入するのが安全です。**

### 再開

```bash
# 実験を再開（suffix の既定は "resumed"）
make resume_exp name=<exp_id_or_name>

# suffix を指定して再開
make resume_exp name=<exp_id_or_name> suffix=retry
```

元の実験の設定・コードが**新しいIDの実験としてコピー**されます（元は消えません）。
再開元の情報は新しい実験の `metadata.yaml` に `resumed_from_exp` /
`resume_checkpoint_dir` として記録されます。チェックポイントの実際の読み込みパスは
`config.yml` に自分で書いてください。

元の実験で `#SBATCH --dependency=` が有効だった場合は、古い `job_id` を引き継がないよう
自動でコメントアウトされます。

### ログの掃除

```bash
make log_clean       # 散らばった *.out を logs/{exp}/{job_id}/ に整理する
make clean_failed    # 不要になったジョブログを削除する
```

`make clean_failed` が消すのは次の3つです。

1. `FAILED` / `TIMEOUT` / `OUT_OF_MEMORY` / `NODE_FAIL` / `CANCELLED` で終わったジョブのログ
2. `RUNNING` のまま放置されたゾンビログ（**`squeue` / `qstat` にそのジョブが
   存在しないことを確認できた場合のみ**。確認できない場合は安全側に倒して残します）
3. 対応する `experiments/` が無い孤立ログ

> [!WARNING]
> 3 の判定は「`logs/{名前}` に対応する `experiments/{名前}` があるか」だけを見ます。
> `experiments/` を**手動で** `mv` や `rm` した場合、そのログは次回の
> `make clean_failed` で消えます。改名・失敗マークは必ず `make rename_exp` /
> `make mark_fail` 経由で行ってください（これらは `logs/` も一緒に動かします）。

---

## Slack通知

ジョブの状態変化時にSlack通知が届きます。

| タイミング | 通知 |
|---|---|
| ジョブ開始 | 🚀 STARTED |
| 正常終了 | ✅ FINISHED |
| 失敗 | ❌ FAILED |
| タイムアウト | ❌ FAILED (TIMEOUT) |
| キャンセル・割り込み | ⚡ INTERRUPTED |

通知のオン・オフは環境変数で制御できます。

```bash
export SLACK_NOTIFY_ON_START=1
export SLACK_NOTIFY_ON_FINISH=1
export SLACK_NOTIFY_ON_FAIL=1
```

---

## ジョブログ

ジョブ終了後、`logs/{exp_name}/{job_id}/` に以下が保存されます。

| ファイル | 内容 |
|---|---|
| `slurm.out` | ジョブの標準出力（Slurm時は標準エラーも含む） |
| `pbs.err` | PBS (qsub) 時の標準エラー出力 |
| `run_metadata.yaml` | ジョブのメタデータ・最終ステータス |
| `command.sh` | 実行されたコマンド |

`run_metadata.yaml` のステータス一覧：

```
RUNNING / COMPLETED / FAILED / TIMEOUT / CANCELLED / OUT_OF_MEMORY / NODE_FAIL
```

---

## Jupyter

```bash
make jupyter p=<partition> mem=<memory>

# 例
make jupyter p=small-david01 mem=32g
```

`env/env.sif` と `.venv` の両方が必要です。どちらか欠けている場合は投入したジョブがその場で失敗し、次に実行すべきコマンド（`make build_env` / `make uv_sync`）を表示します。

---

## ディスクの後片付け

ジョブは高速化のため、ノード付属SSD の `/scratch/$USER/{exp_name}_{job_id}/` を
作業領域に使います（`run_slurm.sh` の `USE_LOCAL_SSD_INPUT` / `USE_LOCAL_SSD_OUTPUT`、
既定でどちらも `1`）。出力はジョブ終了時に `outputs/` へ自動で回収されますが、

> [!IMPORTANT]
> **`/scratch` 側のディレクトリは自動削除されません。** 誤削除を避けるため、
> `slurm.out` の末尾に「消していないので手で消してください」という警告を出すだけに
> 留めています。放置するとノードのSSDが埋まるので、不要になったら自分で消してください。

```bash
ls  /scratch/$USER/           # 残っているものを確認
rm -rf /scratch/$USER/<exp_name>_<job_id>
```

また `USE_LOCAL_SSD_INPUT=1` は**毎ジョブ `data/` を丸ごとコピーします**。
データセットが大きい場合や、必要なのが一部だけの場合は `0` にして NFS から
直接読むほうが速いことがあります。

---

## 困ったときは

| 症状 | 原因と対処 |
|---|---|
| `make create_exp` が `❌ Could not infer partition owner from path:` で止まる | リポジトリが対応パス（[セットアップ 0-(b)](#0-事前に確認すること)）の外にあります。`tools/create_exp.sh` の `_resolve_partition()` に自分の環境を追記してください |
| `type runx` が `not found` | シェルヘルパーが読み込まれていません。[セットアップ 4](#4-シェルヘルパーの設定) をやり直すか、新しいターミナルを開いてください |
| `runx` が `❌ Git working tree is dirty.` で止まる | 未コミットの変更があります。コミットしてから投入してください（急ぐなら `runx --allow-dirty`） |
| ジョブが一瞬で FAILED。`slurm.out` に `unrecognized arguments` | `RUN_COMMAND` や `GRID_ARGS` で渡している引数が `experiment.py` の `parse_args()` に `add_argument()` されていません。両方を一致させてください |
| `slurm.out` に `⚠️ FALLBACK: running on the HOST` | `env/env.sif` が見つからず、コンテナ外で実行されています。結果の再現性が保証されないので `make build_env p=<partition>` をやり直してください |
| `slurm.out` に `.venv/bin/activate: No such file or directory` | `make uv_sync` が未実行か失敗しています。`ls .venv/bin/python` で確認してください |
| ジョブが `squeue` で `PENDING` のまま進まない | 要求リソースが大きすぎます。`squeue -u $USER --start` で開始見込みを、`sinfo` で空き状況を確認し、`--time` / `--mem` / `--gres` を減らしてください |
| ジョブは `COMPLETED` なのに `outputs/` が空 | 既に `completion.json` が `completed` の variant は再実行がスキップされます（`slurm.out` に `Run is already completed:` と出ます）。やり直すには該当ディレクトリを削除してから再投入してください |
| Slack 通知が来ない | `echo $SLACK_WEBHOOK_URL` が空でないか、`jq` が入っているかを確認してください。通知の失敗はジョブ自体を止めず、`slurm.out` に警告だけ残ります |

ジョブの状態を調べる順番は **`run_metadata.yaml` の `status` / `fail_reason` を見る →
必要な箇所だけ `slurm.out` を読む** が最短です。

```bash
cat  logs/<exp_name>/latest/run_metadata.yaml   # まず status と fail_reason
less logs/<exp_name>/latest/slurm.out           # 次に詳細（終了後はここに移動済み）
cat  logs/<exp_name>/latest/command.sh          # 実際に実行されたコマンド
```

---

## 更新履歴 (Change Log)

- **2026-09-11 (READMEの刷新)**: セットアップ・運用手順を実態に合わせて書き直し、Slurm に不慣れでも追えるようにした。用語説明と各ステップの成功確認（`✅ 確認`）、対応パス外に clone すると `make create_exp` が止まる制約、`runx` の dirty チェックが未追跡ファイルを見ない点、`/scratch` が自動削除されない点、`cancelx --all` の影響範囲を明記。`make preflight` / `make log_clean` / `make clean_failed` を手順に追加し、「困ったときは」の索引表を新設。
- **2026-09-11 (稼働中ジョブを壊す経路の修正)**: ログ掃除・キャンセル・リネームが、実行中のジョブや無関係なジョブの状態を破壊していた経路をまとめて修正。(1) `make clean_failed` が `RUNNING` のゾンビ判定を `run_metadata.yaml` の mtime だけで行っていたため、既定の `--time=196:00:00` のような長時間ジョブが**稼働中にログごと削除**されていた（同ファイルは起動時に1回書かれるだけで実行中は更新されない）。`squeue`/`qstat` にジョブが無いと確認できた場合のみ削除するようにし、実装を `tools/clean_failed_logs.sh` に切り出した。(2) `cancelx` のメタデータ検索が前方一致だったため、`cancelx 1234` が `12345` 等**無関係な稼働中ジョブのステータスまで `CANCELLED` に書き換えて**いた。完全一致と `{job_id}_` のみに限定。(3) `make mark_fail` / `make rename_exp` が `logs/` を置き去りにし、孤立ログとして `make clean_failed` に消されていたため、`logs/` も併せて改名するようにした。(4) `templates/run_slurm.sh` の `RUN_COMMAND` が `--config config.yml` を渡す一方 `templates/experiment.py` の `parse_args()` は引数を1つも定義しておらず、**`make create_exp` 直後の `runx` が argparse の exit 2 で必ず失敗**していたため、投入側から `--config` を外した（`config.yml` は `experiment.py` が `__file__` の隣から自動で読む）。(5) パーティション所有者の解決に `/workspace/filesrv02` を追加。
- **2026-09-11 (メタデータ書き込みの統一)**: `run_metadata.yaml` の読み書きを `scripts/metadata.sh` に集約し、**PyYAML依存を廃止**。従来は `python3` + `yaml.safe_dump` で書く経路と `printf` で直書きする経路が混在し、ホストの `python3` に PyYAML がある保証が無いままジョブのライフサイクル管理がそれに依存していた（`update_metadata` が失敗すると `status: RUNNING` のまま残り、`make clean_failed` の削除対象になりうる状態だった）。`tools/cancel_job.sh` に `set -euo pipefail` を追加。
- **2026-09-11 (依存の分野別整理)**: `pyproject.toml` を「分野を問わないベース」と `[project.optional-dependencies]`（`llm` / `chem` / `nlp`）に再構成。`make uv_sync p=<partition> extras="llm chem"` で追加できる。`torch` を cu128 に統一して `env/env.def` のベースイメージ (CUDA 12.8.1) と揃え、テンプレート自身が使う `pyyaml` を明示的に宣言。未使用だった `slackweb`・`pynvml`・`cython`・`fastparquet`・`notebook`・`dask`・`matplotlib-venn` を削除。`uv.lock` はテンプレートには同梱せず、派生プロジェクト側で解決する。
- **2026-09-11 (SIFビルドの同梱)**: Apptainerイメージのビルド定義 `env/env.def` と `make build_env p=<partition>` を追加。`SIF_PATH` の解決を `scripts/sif_env.sh` に集約し （export済み → `.env` → `env/env.sif`）、`make uv_sync` / `make jupyter` が イメージ不在を検知してその場で失敗するようにした（従来は偽の成功メッセージを出していた）。`make setup` を冪等化。
- **2026-06-19 (qsub対応)**: PBS (qsub/Miyabi) 環境への対応を追加。スケジューラ自動検知、PBS用ヘッダー・パーティション選択・ログ回収・ジョブキャンセル・依存指定の追加。
- **2026-06-16 (Slackアレイ通知)**: Slackへのアレイジョブ通知機能、およびリモート環境からのデータ収集・送信スクリプトを追加。
- **2026-06-13 (メタデータ分離)**: 設定ファイル (`config.yml`) と実行メタデータの分離、および resume/create ツールのアップデート。
- **2026-06-13 (出力ディレクトリ)**: 出力ディレクトリ構造の整理とベース設定の追加。
