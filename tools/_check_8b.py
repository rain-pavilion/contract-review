"""核对 LegalOne-8B 权重是否完整（分片按 index 的 weight_map 反查）。"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DIR = ROOT / "zhilv_qiling_v2" / "models" / "pretrained" / "legalone-8b"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not DIR.exists():
        print("目录不存在：", DIR)
        return 2

    index_path = DIR / "model.safetensors.index.json"
    if not index_path.exists():
        print("缺 index.json")
        return 2
    idx = json.loads(index_path.read_text(encoding="utf-8"))
    need = sorted(set((idx.get("weight_map") or {}).values()))
    print(f"index 声明 {len(need)} 个分片：")
    total = 0.0
    missing = []
    for name in need:
        p = DIR / name
        if p.exists():
            gb = p.stat().st_size / 2 ** 30
            total += gb
            print(f"  [OK]   {name}  {gb:.2f} GB")
        else:
            missing.append(name)
            print(f"  [MISS] {name}")
    print(f"合计 {total:.2f} GB ｜ 缺失 {len(missing)} 个")

    parts = [p.name for p in DIR.glob("*.part")]
    print("残留 .part：", parts or "无")

    for fname in ("config.json", "tokenizer.json", "tokenizer_config.json",
                  "generation_config.json", "vocab.json", "merges.txt",
                  "special_tokens_map.json", "added_tokens.json"):
        p = DIR / fname
        print(f"  {'有' if p.exists() else '缺'} {fname}")

    cfg = json.loads((DIR / "config.json").read_text(encoding="utf-8"))
    print("架构", cfg.get("architectures"), "层数", cfg.get("num_hidden_layers"),
          "hidden", cfg.get("hidden_size"), "dtype", cfg.get("torch_dtype"),
          "vocab", cfg.get("vocab_size"))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
