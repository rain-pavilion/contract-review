"""把 BGE-M3 密集通道改为**默认关闭**（opt-in）。

事实：接上 BGE 当 dense 通道后，强金标准 Recall@5 从 0.628 掉到 0.167（见
`reports/gpu_upgrade_bge_m3.json`）。它是负结果，不能默认生效 —— 但之前 `_build_bge()`
是无条件调用，导致后续所有评测都悄悄换成了 BGE，论文里的 0.6944 复现不出来。

改法：`Config.use_bge = False`，只有显式开（配置字段或 `ZLQL_USE_BGE=1`）才加载 BGE。
"""

from __future__ import annotations

import pathlib

cfg_path = pathlib.Path("zhilv_qiling_v2/config.py")
cfg = cfg_path.read_text(encoding="utf-8")

anchor = '''    store_path: str = ""'''
addition = '''    use_bge: bool = False
    """密集通道是否用 BGE-M3 向量；默认关（实测强金标准 Recall@5 0.628 -> 0.167，属负结果）。"""

    store_path: str = ""'''
if "use_bge" not in cfg:
    assert anchor in cfg
    cfg = cfg.replace(anchor, addition, 1)
    cfg_path.write_text(cfg, encoding="utf-8")
    print("config.py 加 use_bge")
else:
    print("config.py 已有 use_bge")

# ---- env 里读 ZLQL_USE_BGE ----
env_anchor = "ZLQL_SEED"
if env_anchor in cfg:
    print("config_from_env 里含 ZLQL_SEED，稍后手工确认 use_bge 的读取")

# ---- retrieve.py：仅在 use_bge 时构建 ----
ret_path = pathlib.Path("zhilv_qiling_v2/retrieve.py")
ret = ret_path.read_text(encoding="utf-8")
old = "        self._build_bge()"
if old in ret:
    new = (
        "        if getattr(config, \"use_bge\", False):\n"
        "            self._build_bge()\n"
        "        else:\n"
        "            self.bge_ready = False\n"
        "            self.bge_matrix = None\n"
        "            self.bge_meta = {}"
    )
    ret = ret.replace(old, new, 1)
    ret_path.write_text(ret, encoding="utf-8")
    print("retrieve.py 已把 BGE 改为 opt-in")
else:
    print("retrieve.py 没找到 _build_bge() 调用，检查一下")
