---
name: スクリプトの配置場所
description: notebooks/src/configs はプロジェクトルートではなく exps/expXX/expXX_XX/ 配下に置く
type: feedback
---

すべてのスクリプト（ノートブック・src モジュール・configs）は `exps/expXX/expXX_XX/` 配下に配置する。プロジェクトルートの `src/`・`notebooks/`・`configs/` には実装しない。

**Why:** 各 child-exp は自己完結した構造を持つべきであり、サンプルコード（`sample/`）もこのパターンに従っている。

**How to apply:** 新規 exp・child-exp を実装するときは、必ず以下の構造に従う。
```
exps/expXX/expXX_XX/
├── configs/config.py
├── src/
├── notebooks/
└── output/
```
ノートブックの `sys.path.append(os.path.abspath('..'))` で `expXX_XX/` をパスに追加することで `from configs.config import *` と `from src.X import Y` が解決される。
