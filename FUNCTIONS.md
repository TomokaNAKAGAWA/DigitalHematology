# template-daily-research-experiments 関数・スクリプト一覧


## 1. `.bashrc.d/` — シェル関数(`source`すると使えるようになる)

### `0-bootstrap.sh`

| 関数 | 内容 |
|---|---|
| `_is_interactive_shell()` | 現在のシェルが対話的か(`$-`に`i`が含まれるか)を判定するだけのヘルパー |

### `1-experiments.sh` — 実験の移動・投入・キャンセル

| 関数 | 内容 |
|---|---|
| `groot()` | Gitリポジトリのルートへ`cd`する |
| `_get_dynamic_root()` | 実行時点のGitルートを取得する内部ヘルパー(`groot`とは別に、他関数から都度呼ぶ用) |
| `lsx()` | `experiments/`配下の最新10件を`ID名`形式で一覧表示 |
| `cdx [id]`  | `experiments/{id}_*/`へ移動。ID省略時は最新へ移動 |
| `runx [id] [--allow-dirty]` | 実験をジョブ投入する本体。git dirty checkを行い、`run_slurm.sh`（完全な#SBATCHスクリプト）をそのまま`sbatch`する。リソース/partition/array-seq切り替え/依存ジョブ指定（`#SBATCH --dependency=`）は`run_slurm.sh`側の責務 |
| `cancelx <job_id> [reason]` | `tools/cancel_job.sh`を呼び出してジョブをキャンセル(内部で`scancel`/`qdel`) |

### `2-git.sh` — Git初回セットアップ・daily branch運用(Pattern2)


| 関数 | 内容 |
|---|---|
| `_setup_git_config()` | `mizuno-group`への直接コミットを禁止するpre-commit hookの設定のみに縮小 |
| `gstart()` | mainを最新化し、`daily/YYMMDD`ブランチを作成/checkoutする |
| `gpush()` | `daily/YYMMDD`をpushし、mainへ`--no-ff`マージしてpush。`gpush -gh`ならローカルマージの代わりに`gh pr create`でPRを作成し、確認後`gh pr merge`でマージする（要`gh` CLI・`gh auth login`） |


## 2. `lib/` — Pythonライブラリ関数

### `lib/output_utils.py`

| 関数 | 内容 |
|---|---|
| `sanitize_variant_key(variant_key)` | `variant_key`に`/`が含まれる場合、最後の要素だけを取り出して1階層のディレクトリ名に正規化する(`review-exp`/`roadmap`のglobパターンが1階層構造を前提にしているため) |
| `get_run_dir(project_root, script_path, variant_key)` | `outputs/{exp_name}/{variant_key}/`を返す。`completion.json`の`status`が`completed`なら`sys.exit(0)`で再実行をガードし、`guard_skipped_at`を記録する(completedガード) |
| `write_run_metadata(run_dir, **kwargs)` | `completion.json`に`job_id`/`array_task_id`/`started_at`/`status=running`等のメタデータを書き込む |
| `complete_run(run_dir)` | `completion.json`の`status`を`completed`に更新し、`completed_at`を記録する |

## 3. `tools/` — 個別コマンドの実体(Makefileから呼ばれる)

| スクリプト | 呼び出し元 | 内容 |
|---|---|---|
| `create_exp.sh` | `make create_exp name=<name>` | 4桁ID自動採番で実験ディレクトリ作成(git操作は行わない。daily/YYMMDD方式のためブランチは切らない)。`run_slurm.sh`コピー時にpartition/signal marginを自動計算して焼き込む |
| `mark_failed.sh` | `make mark_fail name=<n> reason=<r>` | ディレクトリ名に`_FAILED_<date>_<reason>`を付与 |
| `resume_exp.sh` | `make resume_exp name=<n> [suffix=<s>]` | 元の実験の設定・コードをコピーし新IDで再作成、再開元をconfig.ymlに記録。元でアクティブだった`#SBATCH --dependency=`は古いjob_idを引き継がないよう無効化される |
| `rename_exp.sh` | `make rename_exp name=<n> new=<new>` | 実験ディレクトリのリネーム |
| `cancel_job.sh` | `cancelx <job_id>` | `scancel`/`qdel`のスケジューラ差異を吸収してキャンセル。`run_metadata.yaml`の更新は`scripts/metadata.sh`経由（PyYAML不要）。`--all`では1件失敗しても残りを続行する |
| `uv_sync.sh` | `make uv_sync p=<partition> [extras="llm chem"]` | `sbatch`投入され、コンテナ内で`uv sync`して`.venv`を作るジョブ本体。SIF不在なら`require_sif`で即失敗する。`extras`は位置引数で受け取り`--extra <name>`に展開する（`extras=all`なら`--all-extras`）。`bash -c`文字列に埋め込むため、extra名は英数字・`_`・`-`のみに制限して検証する |
| `start_jupyter.sh` | `make jupyter p=<p> mem=<m>` | `sbatch`投入され、Jupyter Serverをcompute node上に起動 |
| `first_setup.sh` | `make setup` | 初期セットアップ一式（`.env`と各ディレクトリ作成。冪等） |
| `build_env.sh` | `make build_env p=<partition>` | `sbatch`投入され、`env/env.def`から`env/env.sif`をビルドするジョブ本体。`env/env.sif.new`に作って成功時のみ差し替えるため、失敗しても既存イメージは壊れない |

## 4. `scripts/` — ジョブ実行・監視の内部スクリプト

| スクリプト | 内容 |
|---|---|
| `slurm_entry.sh` | 実際にジョブ内で呼ばれるエントリーポイント。実行・ログ回収・ステータス判定を担う |
| `sif_env.sh` | `SIF_PATH`の解決規則。`tools/uv_sync.sh`・`tools/start_jupyter.sh`・`scripts/slurm_entry.sh`から`source`される（下表参照） |
| `metadata.sh` | `logs/{exp}/{job_id}/run_metadata.yaml`の読み書き。`scripts/slurm_entry.sh`・`tools/cancel_job.sh`から`source`される。**PyYAMLを使わないprintf実装**（下表参照） |
| `preflight_check.py`(`make preflight`の実体、`python3 -m scripts.preflight_check`として呼ばれる) | 下表参照 |
| `notify_slack.sh` | Job IDから色を生成してSlackへ通知するロジック |

### `scripts/sif_env.sh` の関数

| 関数 | 内容 |
|---|---|
| `_sif_path_from_env_file(project_root)` | `<project_root>/.env`から`SIF_PATH`の行だけをgrepで読む。`.env`は`source`しない（任意のコード実行を避けるため）。先頭の`~`のみ自前で展開する |
| `resolve_sif_path(project_root)` | `SIF_PATH`を解決して`export`する。優先順位は 1) 既に`export`済みの値 2) `.env` 3) `<project_root>/env/env.sif`。存在検証はしない |
| `require_sif(project_root)` | `resolve_sif_path`した上で`apptainer`コマンドとイメージファイルの存在を検証する。使えない場合は次に実行すべきコマンド（`make build_env p=<partition>`）を示して非ゼロで返る |

### `scripts/metadata.sh` の関数

`run_metadata.yaml`はフラットなスカラーのマップしか持たないため、YAMLパーサ(PyYAML)は
使わない。以前は`python3` + `yaml.safe_dump`で書く経路と`printf`で直書きする経路
(`_bootstrap_fail`)が混在しており、かつホストの`python3`にPyYAMLがある保証が無いまま
ジョブのライフサイクル管理がそれに依存していた(シグナルハンドラ内でpythonサブプロセスを
起動する形にもなっていた)。値は常にダブルクォートで囲むため、`:`や`#`を含む
`fail_reason`でもYAMLとして妥当なまま保たれる。

| 関数 | 内容 |
|---|---|
| `_metadata_quote(value)` | 値をYAMLのダブルクォート文字列にする(`\`と`"`をエスケープ) |
| `_metadata_unquote(value)` | ダブルクォート文字列を元に戻す |
| `metadata_init(file, key, value, ...)` | ファイルを新規作成する(既存内容は破棄)。キー/値のペアを可変個受け取る |
| `metadata_set(file, key, value)` | キーがあれば置換、なければ追記する。既存行の順序は保たれる |
| `metadata_get(file, key)` | 値を標準出力に返す。キーが無ければ非ゼロで返る |

### `scripts/preflight_check.py` の関数

| 関数 | 内容 |
|---|---|
| `get_latest_experiment()` | `experiments/`配下で最新の実験ディレクトリ(4桁ID)を取得 |
| `check_config_experiment_match(exp_path)` | `config.yml`のトップレベルキーと`experiment.py`内の参照キーの不整合を検出 |
| `check_grid_size(exp_path)` | `run_slurm.sh`の`GRID_VALUES`の直積数を計算し、上限(50件)超過を警告。`GRID_ARGS`定義済みなのに`RUN_MODE`が`array`/`seq`でない場合、および`RUN_MODE="array"`時に`#SBATCH --array=0-N`のNが直積数と不一致の場合もBlockerにする |
| `check_direct_writes(exp_path)` | `experiment.py`内で`outputs/`/`data/`への直接書き込み(`to_parquet`/`to_csv`/`open`のハードコードパス)を検出 |
| `main()` | 上記チェックを順に実行し、全てOKなら`✅ preflight OK`、NGがあれば非ゼロ終了 |

## 5. 各関数を素で書いた場合の相当コマンド

`.bashrc.d/1-experiments.sh`・`.bashrc.d/2-git.sh`の関数は、内部で以下のコマンドを
実行しているだけのラッパーである。agentrun capsule内のhook(`check_sbatch_forbidden.py`・
`git_branch_guard.py`)はラッパー関数名ではなく最終的に実行される生のコマンドを見て
許可/拒否を判定しているため、「なぜこの関数呼び出しが許可/拒否されたか」を理解する
助けとして対応関係を示す。

| 関数 | 素で書いた場合の相当コマンド |
| ---- | ---- |
| `groot` | `cd $(git rev-parse --show-toplevel)` |
| `lsx` | `ls -d experiments/*/ \| sort -r \| head -n 10`(実際はID/名前だけに整形して表示) |
| `cdx [id]` | ID指定時: `cd experiments/{4桁ID}_*/`。省略時: `cd $(ls -d experiments/*/ \| sort -r \| head -n1)` |
| `runx [id] [--allow-dirty]` | `git diff --quiet && git diff --cached --quiet`(dirtyチェック。`--allow-dirty`で省略)→`mkdir -p logs/{exp_name}`→**`sbatch experiments/{exp_name}/run_slurm.sh`**→`echo <job_id> > outputs/{exp_name}/latest_job_id.txt`。**hookが実際に見ているのは太字の`sbatch`行**であり、`runx`はそこへの経路を用意しているだけ |
| `cancelx <job_id> [reason]` | `run_metadata.yaml`の`status`を`CANCELLED`に更新→Slack通知(`notify_fail_fast`)→`scancel <job_id>`(PBSなら`qdel <job_id>`) |
| `gstart`(host専用。agentrun capsule内では常にブロック) | `git checkout main && git pull origin main`→`git checkout -b daily/YYMMDD`(既存なら`git checkout daily/YYMMDD`)→(originに同名ブランチがあれば`git pull --ff-only origin daily/YYMMDD`) |
| `gpush`(host専用。agentrun capsule内では常にブロック) | `git push origin daily/YYMMDD`→`git checkout main && git pull origin main`→`git merge --no-ff daily/YYMMDD -m "merge daily/YYMMDD into main"`→`git push origin main`(`-gh`オプション時はローカルマージの代わりに`gh pr create`→確認→`gh pr merge`) |

`groot`/`lsx`/`cdx`/`cancelx`はジョブ投入を伴わないため、この対応表の中で
agentrun capsule内のhookが実際に制限対象にするのは`runx`(内部の`sbatch`行)と
`gstart`/`gpush`(内部の`git checkout main`/`git push`/`git merge`行)だけである。
