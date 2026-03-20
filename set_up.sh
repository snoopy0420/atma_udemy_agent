# プロジェクト開始時に使用する
# ターミナルを再起動すると環境変数のこの変更が確認できる。
# windowをリロードするとソース管理も反映される。
# vscodeのフォルダもatma_udemyを開きカーネル(.venv/bin/python)を作成する。

# bashスクリプトの安全装置3点セット
set -euo pipefail

# uvのセットアップ
## インストール
# curl -LsSf https://astral.sh/uv/install.sh | sh
# ## パスを通す
# export PATH=/root/.local/bin:$PATH
# ## uvのキャッシュを/workspace配下にする
# export UV_CACHE_DIR=/workspace/.cache/uv
# mkdir -p /workspace/.cache/uv

# gitのセットアップ
# git config --global user.email "runpod@example.com"
# git config --global user.name "runpod"

# git clone
cd /workspace 
# git clone {もとにするgitリポジトリurl} # 要修正

# gitプロジェクトの初期化
# 事前に新しいプロジェクト用のgitリポジトリを作成しておく
mv -n 旧ディレクトリ名 新ディレクトリ名
cd /workspace/atma_udemy_agent # 要修正
rm -rf .git
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/snoopy0420/atma_udemy_agent.git # 要修正
git branch -M main
git push -u origin main

# uvの初期化
uv init
uv add --dev nbstripout ipykernel
uv python install 3.12.12 # 要修正

# ipynbの出力をコミット時に自動削除する
uv run nbstripout --install

# 仮想環境をactive
source .venv/bin/activate





