"""预训练权重定位：不写死某个仓库名，按"真的能加载"来判断。

背景：中文 BERT 有很多近义仓库（`hfl/chinese-roberta-wwm-ext`、`bert-base-chinese`、
`chinese-roberta-wwm-ext-large`…），不同镜像站的可达性也不同。写死一个名字，
换台机器就得改代码。这里改成按目录探测：

- 目录里有 `config.json`；
- 且有 `model.safetensors` / `pytorch_model.bin`，体积大于 50 MB（排除只有 tokenizer 的半成品）；
- 按 `PREFERENCE` 的顺序取第一个满足条件的。

`resolve()` 只做文件系统检查，不 import torch，因此可以被任意模块安全调用。
"""

from __future__ import annotations

import pathlib

from .config import MODEL_DIR

PRETRAINED_ROOT = MODEL_DIR / "pretrained"

#: 优先顺序：roberta 效果略好，bert-base 更小更常见，其余作兜底
PREFERENCE: tuple[str, ...] = (
    "chinese-roberta-wwm-ext",
    "chinese-roberta-wwm-ext-large",
    "bert-base-chinese",
    "chinese-bert-wwm-ext",
    "bge-m3",
)

WEIGHT_FILES: tuple[str, ...] = ("model.safetensors", "pytorch_model.bin")
MIN_WEIGHT_BYTES = 50 * 1024 * 1024


def _has_weights(directory: pathlib.Path) -> bool:
    if not (directory / "config.json").exists():
        return False
    for name in WEIGHT_FILES:
        target = directory / name
        if target.exists() and target.stat().st_size >= MIN_WEIGHT_BYTES:
            return True
    # 分片权重（model-00001-of-00002.safetensors）
    shards = sorted(directory.glob("model-*-of-*.safetensors"))
    return bool(shards)


def candidates() -> list[pathlib.Path]:
    """按优先顺序返回候选目录（只返回存在的）。"""
    found: list[pathlib.Path] = []
    for name in PREFERENCE:
        path = PRETRAINED_ROOT / name
        if path.is_dir():
            found.append(path)
    if PRETRAINED_ROOT.is_dir():
        for path in sorted(PRETRAINED_ROOT.iterdir()):
            if path.is_dir() and path not in found:
                found.append(path)
    return found


def resolve(name: str | None = None) -> pathlib.Path | None:
    """返回可用的权重目录；都不可用时返回 None（调用方负责报错或降级）。"""
    if name:
        path = PRETRAINED_ROOT / name
        return path if _has_weights(path) else None
    for path in candidates():
        if _has_weights(path):
            return path
    return None


def describe() -> dict[str, object]:
    """诊断用：每个候选目录是否齐备、权重多大。"""
    rows = []
    for path in candidates():
        weights = [path / name for name in WEIGHT_FILES if (path / name).exists()]
        size = sum(item.stat().st_size for item in weights if item.exists())
        rows.append(
            {
                "name": path.name,
                "ready": _has_weights(path),
                "weight_mb": round(size / 1048576, 1),
            }
        )
    return {"root": str(PRETRAINED_ROOT), "candidates": rows, "resolved": (resolve() or "").__str__()}
