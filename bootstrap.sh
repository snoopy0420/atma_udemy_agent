# podのリスタート時に実行する

# bashスクリプトの安全装置3点セット
set -euo pipefail

# gitのセットアップ
git config --global user.email "runpod@example.com"
git config --global user.name "runpod"
uv run nbstripout --install
