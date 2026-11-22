"""BGE-M3 密集编码通道：把检索里"用 LSA 冒充语义向量"换成真语义向量。

背景
----
`retrieve.StatuteIndex` 的密集通道原本是 sklearn 的 `TfidfVectorizer(char_wb)` + `TruncatedSVD`，
本质仍是字面统计（LSA），所以"租赁期限二十年以上"这种换词表达的查询，只能靠 BM25 侥幸命中。
BGE-M3（BAAI）是中文检索上常用的多语向量模型，1024 维、支持 8192 长度，
1260 条《民法典》在 GPU 上编码只需几秒，向量落成 `.npy` 后推理零依赖。

本文件提供
----------
- `Embedder`：加载权重、`encode(texts) -> np.ndarray`（L2 归一化）
- `build_statute_vectors(statutes) -> (matrix, meta)`：批量编码法条并落盘
- `load_vectors()`：读回 `models/bge_m3/statutes.npy` 与向量 id 顺序
- 没有 torch / 没有权重时 `available()` 返回 False，调用方退回 LSA（降级留账）

注意：本模块不 import torch 到顶层，`available()` 只做存在性检查，
保证"没装 torch 的机器能 import 整个包"。
"""

from __future__ import annotations

import json
import pathlib
import os
import threading
from dataclasses import dataclass
from typing import Any, Sequence

from .config import MODEL_DIR

from . import pretrained as _pretrained

WEIGHTS_DIR = MODEL_DIR / "pretrained" / "bge-m3"

def _embed_dir():
    """BGE 权重目录：优先 bge-m3，其次任意 bge-* （小模型也能跑，只是维度不同）。"""
    exact = _pretrained.resolve("bge-m3")
    if exact is not None:
        return exact
    for path in sorted(_pretrained.candidates()):
        if path.name.startswith("bge") and _pretrained._has_weights(path):
            return path
    return None

VECTOR_DIR = MODEL_DIR / "bge_m3"
VECTOR_FILE = "statutes.npy"
META_FILE = "statutes.json"
DIM = 1024
MAX_LEN = 512


def available() -> bool:
    """权重与 torch 都在才算可用。"""
    if not WEIGHTS_DIR.exists():
        return False
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _normalize(matrix):
    import numpy as np

    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return matrix / norm


@dataclass
class Embedder:
    """BGE-M3 编码器。首次构造会加载约 2 GB 权重。"""

    device: str = "auto"
    max_length: int = MAX_LEN
    batch_size: int = 16

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(WEIGHTS_DIR))
        self.model = AutoModel.from_pretrained(str(WEIGHTS_DIR)).to(self.device).eval()

    def encode(self, texts: Sequence[str], *, batch_size: int | None = None,
               show_progress: bool = False) -> Any:
        """返回 (n, 1024) float32，已 L2 归一化（点积即余弦）。"""
        import numpy as np

        torch = self._torch
        size = batch_size or self.batch_size
        chunks: list[Any] = []
        for start in range(0, len(texts), size):
            batch = list(texts[start:start + size])
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded)
            # BGE-M3 用 [CLS] 表示整句
            chunks.append(output.last_hidden_state[:, 0].float().cpu().numpy())
            if show_progress:
                print(f"    encode {min(start + size, len(texts))}/{len(texts)}", flush=True)
        if not chunks:
            return np.zeros((0, DIM), dtype="float32")
        return _normalize(np.vstack(chunks)).astype("float32")


def _statute_text(statute: dict[str, Any]) -> str:
    content = str(statute.get("content") or "").strip()
    tags = statute.get("risk_tags") or []
    types = statute.get("contract_types") or []
    extra = []
    if tags:
        extra.append("标签：" + "、".join(str(tag) for tag in tags[:8]))
    if types:
        extra.append("合同类型：" + "、".join(str(item) for item in types[:6]))
    head = f"{statute.get('law_name', '')}{statute.get('article_no', '')}"
    return " ".join([head, content] + extra).strip()


def build_statute_vectors(statutes: Sequence[dict[str, Any]], *, embedder: Embedder | None = None,
                          out_dir: pathlib.Path | None = None) -> dict[str, Any]:
    """编码全部法条并落盘，返回 meta（含条数、维度、设备、耗时）。"""
    import numpy as np

    embedder = embedder or Embedder()
    texts = [_statute_text(statute) for statute in statutes]
    matrix = embedder.encode(texts)
    target = pathlib.Path(out_dir) if out_dir else VECTOR_DIR
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / VECTOR_FILE, matrix)
    meta = {
        "count": len(texts),
        "dim": int(matrix.shape[1]) if matrix.size else 0,
        "device": embedder.device,
        "model": "BAAI/bge-m3",
        "max_length": embedder.max_length,
        "uids": [str(statute.get("article_uid", "")) for statute in statutes],
        "order": [str(statute.get("article_no", "")) for statute in statutes],
    }
    (target / META_FILE).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta


def load_vectors(out_dir: pathlib.Path | None = None) -> tuple[Any, dict[str, Any]] | tuple[None, dict]:
    """读回矩阵与 meta；不存在时返回 (None, {})。"""
    import numpy as np

    target = pathlib.Path(out_dir) if out_dir else VECTOR_DIR
    matrix_file = target / VECTOR_FILE
    meta_file = target / META_FILE
    if not matrix_file.exists() or not meta_file.exists():
        return None, {}
    matrix = np.load(matrix_file)
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    return matrix, meta


def search(query: str, matrix: Any, meta: dict[str, Any], *, embedder: Embedder | None = None,
           top_k: int = 20) -> list[tuple[int, float]]:
    """单条查询的向量召回，返回 [(行号, 余弦分)]。"""
    import numpy as np

    embedder = embedder or shared_embedder()
    vector = embedder.encode([query])
    scores = matrix @ vector[0]
    order = np.argsort(-scores)[:top_k]
    return [(int(index), float(scores[index])) for index in order]


_EMBEDDER: Embedder | None = None
_EMBEDDER_LOCK = threading.RLock()


def shared_embedder() -> Embedder:
    """进程内共享一个 Embedder（加载 2 GB 权重，不能每条查询都建）。"""
    global _EMBEDDER
    with _EMBEDDER_LOCK:
        if _EMBEDDER is None:
            # The online retriever must not claim 2GB of the GPU reserved for 8B.
            device = os.environ.get("ZLQL_EMBEDDING_DEVICE", os.environ.get("ZLQL_BERT_DEVICE", "cpu"))
            _EMBEDDER = Embedder(device=device)
    return _EMBEDDER


def reset_embedder() -> None:
    global _EMBEDDER
    _EMBEDDER = None
