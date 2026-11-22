"""审查结果持久化与十万份级调度（申报书第 6 条"存储/检索/批量审查"的落地）。

v1 用 qdrant 做向量库，但那是"向量相似检索"，不是"审查结果归档"；v2 只读复用 v1 的
法条索引，因此需要一层自己的存储。这里用 **SQLite 单文件**，理由写在设计里：

1. 零依赖：Python 标准库自带 `sqlite3`，笔记本上开箱即用（申报书要"普通终端可跑"）。
2. 十万份级够用：单表百万行 + 三个索引，`EXPLAIN QUERY PLAN` 走索引，分页查询 1 ms 量级；
   真要更大再换 PostgreSQL，接口不变。
3. 可审计：报告落库后可回放（`get()` 还原成 `ReviewReport`），评审要"结论可溯源"时能直接查。
4. 断点续跑：批量审查按 `contract_id` 主键去重，中断后重跑自动跳过已完成的部分。

表结构：

- `contracts`  一份合同一行：来源、类型、评分、风险数、耗时、文本指纹
- `risks`      一条风险一行：风险码、分类、等级、置信、是否需复核、标签（JSON）
- `citations`  风险 ↔ 法条：法条名、条号、`article_uid`（硬护栏：只存本地库里真实存在的条）
- `runs`       一次批量任务的账：起止时间、处理数、失败数、备注
- `clause_fts` 条款全文检索（FTS5，不可用时自动降级为 LIKE）
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .config import WORKSPACE, Config
from .models import ReviewReport

SCHEMA_VERSION = "2.2"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS contracts (
    contract_id    TEXT PRIMARY KEY,
    source_name    TEXT NOT NULL,
    contract_type  TEXT NOT NULL DEFAULT '',
    score          INTEGER NOT NULL DEFAULT 0,
    risks_count    INTEGER NOT NULL DEFAULT 0,
    high_count     INTEGER NOT NULL DEFAULT 0,
    clauses_count  INTEGER NOT NULL DEFAULT 0,
    elapsed_ms     REAL NOT NULL DEFAULT 0,
    digest         TEXT NOT NULL DEFAULT '',
    text_chars     INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_contracts_score   ON contracts(score);
CREATE INDEX IF NOT EXISTS idx_contracts_created ON contracts(created_at);
CREATE INDEX IF NOT EXISTS idx_contracts_type    ON contracts(contract_type);

CREATE TABLE IF NOT EXISTS risks (
    risk_id          TEXT PRIMARY KEY,
    contract_id      TEXT NOT NULL,
    code             TEXT NOT NULL,
    category         TEXT NOT NULL DEFAULT '',
    severity         TEXT NOT NULL DEFAULT '',
    confidence       REAL NOT NULL DEFAULT 0,
    review_required  INTEGER NOT NULL DEFAULT 0,
    message          TEXT NOT NULL DEFAULT '',
    tags             TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (contract_id) REFERENCES contracts(contract_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_risks_contract ON risks(contract_id);
CREATE INDEX IF NOT EXISTS idx_risks_code     ON risks(code);
CREATE INDEX IF NOT EXISTS idx_risks_severity ON risks(severity);

CREATE TABLE IF NOT EXISTS citations (
    risk_id     TEXT NOT NULL,
    law_name    TEXT NOT NULL,
    article_no  TEXT NOT NULL,
    article_uid TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (risk_id, article_uid, article_no)
);
CREATE INDEX IF NOT EXISTS idx_citations_article ON citations(article_uid);

CREATE TABLE IF NOT EXISTS runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT NOT NULL DEFAULT '',
    processed   INTEGER NOT NULL DEFAULT 0,
    failed      INTEGER NOT NULL DEFAULT 0,
    note        TEXT NOT NULL DEFAULT ''
);
"""

#: 中文检索必须用 trigram 分词器：默认 unicode61 会把整句当成一个 token，
#: "违约责任"这类子串永远匹配不上（实测返回空）。trigram 要求查询 ≥3 字。
_FTS_TRIGRAM = """
CREATE VIRTUAL TABLE IF NOT EXISTS clause_fts USING fts5(
    contract_id UNINDEXED,
    clause_id   UNINDEXED,
    body,
    tokenize = "trigram"
);
"""

_FTS_PLAIN = """
CREATE VIRTUAL TABLE IF NOT EXISTS clause_fts USING fts5(
    contract_id UNINDEXED,
    clause_id   UNINDEXED,
    body
);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class StoredContract:
    """`contracts` 表的一行（给列表页用，不装整份报告）。"""

    contract_id: str
    source_name: str
    contract_type: str
    score: int
    risks_count: int
    high_count: int
    clauses_count: int
    elapsed_ms: float
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "source_name": self.source_name,
            "contract_type": self.contract_type,
            "score": self.score,
            "risks": self.risks_count,
            "high": self.high_count,
            "clauses": self.clauses_count,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "created_at": self.created_at,
        }


@dataclass
class StoreStats:
    contracts: int = 0
    risks: int = 0
    citations: int = 0
    by_severity: dict[str, int] = field(default_factory=dict)
    by_code: dict[str, int] = field(default_factory=dict)
    score_mean: float = 0.0
    score_min: int = 0
    score_max: int = 0
    elapsed_mean_ms: float = 0.0
    pending_review: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "contracts": self.contracts,
            "risks": self.risks,
            "citations": self.citations,
            "by_severity": self.by_severity,
            "by_code": dict(sorted(self.by_code.items(), key=lambda kv: -kv[1])[:20]),
            "score_mean": round(self.score_mean, 2),
            "score_min": self.score_min,
            "score_max": self.score_max,
            "elapsed_mean_ms": round(self.elapsed_mean_ms, 2),
            "pending_review": self.pending_review,
        }


class ReviewStore:
    """审查结果的落库、分页、统计与断点续跑。"""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else WORKSPACE / "data" / "reviews.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        # 优先 trigram（中文子串可检索）；不支持则退回普通 fts5；再不行就只用 LIKE
        self.fts = "trigram"
        for candidate, name in ((_FTS_TRIGRAM, "trigram"), (_FTS_PLAIN, "plain")):
            try:
                self._conn.executescript(candidate)
                self.fts = name
                break
            except sqlite3.OperationalError:
                continue
        else:
            self.fts = ""
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema', ?)", (SCHEMA_VERSION,)
        )
        self._conn.commit()

    # ---- 生命周期 ----
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ReviewStore":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ---- 写入 ----
    def has(self, contract_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM contracts WHERE contract_id = ?", (contract_id,)
        ).fetchone()
        return row is not None

    def save(self, report: ReviewReport, *, digest: str = "", text_chars: int = 0) -> str:
        """写一份报告；同 id 覆盖（重跑同一份不会重复计数）。"""
        elapsed_ms = float((report.stats or {}).get("elapsed_seconds", 0.0)) * 1000.0
        clauses_count = int((report.stats or {}).get("clauses", 0) or 0)
        high = sum(1 for risk in report.risks if risk.severity == "high")
        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO contracts
                   (contract_id, source_name, contract_type, score, risks_count, high_count,
                    clauses_count, elapsed_ms, digest, text_chars, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    report.contract_id,
                    report.source_name,
                    str((report.stats or {}).get("contract_type") or ""),
                    int(report.score),
                    len(report.risks),
                    high,
                    clauses_count,
                    elapsed_ms,
                    digest,
                    text_chars,
                    _now(),
                ),
            )
            self._conn.execute("DELETE FROM risks WHERE contract_id = ?", (report.contract_id,))
            for risk in report.risks:
                self._conn.execute(
                    """INSERT OR REPLACE INTO risks
                       (risk_id, contract_id, code, category, severity, confidence,
                        review_required, message, tags)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        risk.risk_id,
                        report.contract_id,
                        risk.code,
                        risk.category,
                        risk.severity,
                        float(risk.calibrated_confidence or risk.confidence),
                        1 if risk.review_required else 0,
                        risk.message,
                        json.dumps(risk.tags or [], ensure_ascii=False),
                    ),
                )
                for citation in risk.citations:
                    self._conn.execute(
                        """INSERT OR IGNORE INTO citations
                           (risk_id, law_name, article_no, article_uid) VALUES (?,?,?,?)""",
                        (risk.risk_id, citation.law_name, citation.article_no, citation.article_uid),
                    )
        return report.contract_id

    def save_clauses(self, contract_id: str, clauses: Iterable[tuple[str, str]]) -> int:
        """条款正文入全文索引（`clause_id, body`），供"按条款找合同"用。"""
        if not self.fts:
            return 0
        rows = [(contract_id, clause_id, body) for clause_id, body in clauses]
        with self._conn:
            self._conn.execute("DELETE FROM clause_fts WHERE contract_id = ?", (contract_id,))
            self._conn.executemany(
                "INSERT INTO clause_fts(contract_id, clause_id, body) VALUES (?,?,?)", rows
            )
        return len(rows)

    def start_run(self, kind: str, note: str = "") -> int:
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO runs(kind, started_at, note) VALUES (?,?,?)", (kind, _now(), note)
            )
        return int(cursor.lastrowid)

    def finish_run(self, run_id: int, processed: int, failed: int, note: str = "") -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET finished_at=?, processed=?, failed=?, note=? WHERE run_id=?",
                (_now(), processed, failed, note, run_id),
            )

    # ---- 读取 ----
    def list_contracts(
        self,
        *,
        page: int = 1,
        size: int = 20,
        code: str | None = None,
        severity: str | None = None,
        min_score: int | None = None,
        max_score: int | None = None,
    ) -> tuple[list[StoredContract], int]:
        """分页列出合同；带 code/severity 过滤时走 `risks` 子查询。"""
        page = max(1, int(page))
        size = max(1, min(200, int(size)))
        where: list[str] = []
        params: list[Any] = []
        if code or severity:
            sub = ["SELECT contract_id FROM risks WHERE 1=1"]
            if code:
                sub.append("AND code = ?")
                params.append(code)
            if severity:
                sub.append("AND severity = ?")
                params.append(severity)
            where.append(f"c.contract_id IN ({' '.join(sub)})")
        if min_score is not None:
            where.append("c.score >= ?")
            params.append(int(min_score))
        if max_score is not None:
            where.append("c.score <= ?")
            params.append(int(max_score))
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        total = int(
            self._conn.execute(
                f"SELECT COUNT(*) FROM contracts c {clause}", params
            ).fetchone()[0]
        )
        rows = self._conn.execute(
            f"""SELECT c.* FROM contracts c {clause}
                ORDER BY c.created_at DESC, c.contract_id
                LIMIT ? OFFSET ?""",
            params + [size, (page - 1) * size],
        ).fetchall()
        return ([_to_stored(row) for row in rows], total)

    def get_risks(self, contract_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT r.*, (SELECT json_group_array(c.article_no) FROM citations c
                             WHERE c.risk_id = r.risk_id) AS articles
               FROM risks r WHERE r.contract_id = ?
               ORDER BY CASE r.severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                        r.confidence DESC""",
            (contract_id,),
        ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["tags"] = json.loads(item.get("tags") or "[]")
            item["articles"] = json.loads(item.get("articles") or "[]")
            item["review_required"] = bool(item["review_required"])
            out.append(item)
        return out

    def search_clauses(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """按条款正文找合同；FTS 不可用或查询过短时退化为 LIKE。"""
        needle = str(query or "").strip()
        if self.fts and len(needle) >= 3:
            try:
                rows = self._conn.execute(
                    """SELECT contract_id, clause_id,
                              snippet(clause_fts, 2, '[', ']', '…', 12) AS snippet
                       FROM clause_fts WHERE clause_fts MATCH ? LIMIT ?""",
                    (_fts_query(needle), limit),
                ).fetchall()
                if rows:
                    return [dict(row) for row in rows]
            except sqlite3.OperationalError:
                pass
        if self.fts and needle and len(needle) < 3:
            rows = self._conn.execute(
                """SELECT contract_id, clause_id, substr(body, 1, 80) AS snippet
                   FROM clause_fts WHERE body LIKE ? LIMIT ?""",
                (f"%{needle}%", limit),
            ).fetchall()
            if rows:
                return [dict(row) for row in rows]
        rows = self._conn.execute(
            "SELECT contract_id, source_name FROM contracts WHERE source_name LIKE ? LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> StoreStats:
        out = StoreStats()
        out.contracts = int(self._conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0])
        out.risks = int(self._conn.execute("SELECT COUNT(*) FROM risks").fetchone()[0])
        out.citations = int(self._conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0])
        for row in self._conn.execute(
            "SELECT severity, COUNT(*) FROM risks GROUP BY severity"
        ).fetchall():
            out.by_severity[str(row[0])] = int(row[1])
        for row in self._conn.execute("SELECT code, COUNT(*) FROM risks GROUP BY code").fetchall():
            out.by_code[str(row[0])] = int(row[1])
        row = self._conn.execute(
            "SELECT AVG(score), MIN(score), MAX(score), AVG(elapsed_ms) FROM contracts"
        ).fetchone()
        if row and row[0] is not None:
            out.score_mean = float(row[0])
            out.score_min = int(row[1])
            out.score_max = int(row[2])
            out.elapsed_mean_ms = float(row[3] or 0.0)
        out.pending_review = int(
            self._conn.execute(
                "SELECT COUNT(*) FROM risks WHERE review_required = 1"
            ).fetchone()[0]
        )
        return out


def _to_stored(row: sqlite3.Row) -> StoredContract:
    return StoredContract(
        contract_id=str(row["contract_id"]),
        source_name=str(row["source_name"]),
        contract_type=str(row["contract_type"]),
        score=int(row["score"]),
        risks_count=int(row["risks_count"]),
        high_count=int(row["high_count"]),
        clauses_count=int(row["clauses_count"]),
        elapsed_ms=float(row["elapsed_ms"]),
        created_at=str(row["created_at"]),
    )


def _fts_query(text: str) -> str:
    """trigram 分词器下，整串加引号就是子串检索；多段用 OR 连。"""
    tokens = [token for token in str(text).replace('"', " ").split() if token]
    if not tokens:
        return '""'
    return " OR ".join(f'"{token}"' for token in tokens)


def default_store(config: Config | None = None) -> ReviewStore:
    """按配置取默认库路径（`Config.store_path` 为空时落 `data/reviews.sqlite`）。"""
    config = config or Config()
    path = getattr(config, "store_path", "") or None
    return ReviewStore(path)


@dataclass
class BatchResult:
    """批量审查的账：处理了多少、跳过多少、失败原因、耗时。"""

    run_id: int
    processed: int = 0
    skipped: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "processed": self.processed,
            "skipped": self.skipped,
            "failed": len(self.failed),
            "failures": [{"path": path, "reason": reason} for path, reason in self.failed[:20]],
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


def batch_to_store(
    paths: Sequence[Path | str],
    store: ReviewStore,
    config: Config | None = None,
    *,
    limit: int | None = None,
    resume: bool = True,
    on_report: Any = None,
) -> BatchResult:
    """批量审查并落库；`resume=True` 时跳过库里已有的 `contract_id`（断点续跑）。"""
    from . import pipeline

    config = config or Config()
    run_id = store.start_run("batch", note=f"{len(paths)} files")
    result = BatchResult(run_id=run_id)
    started = time.time()
    for path in list(paths)[:limit] if limit else list(paths):
        item = Path(path)
        try:
            document = pipeline.prepare(item.name, path=item, config=config)
            contract = pipeline.build_contract(document, config=config)
            if resume and store.has(contract.contract_id):
                result.skipped += 1
                continue
            report = pipeline.review_document(document, config)
            store.save(
                report,
                digest=contract.contract_id,
                text_chars=len(document.full_text),
            )
            store.save_clauses(
                report.contract_id,
                [(clause.clause_id, clause.text) for clause in document.clauses],
            )
            result.processed += 1
            if on_report is not None:
                on_report(report)
        except Exception as exc:  # noqa: BLE001 —— 单份失败不该中断整批
            result.failed.append((str(item), f"{type(exc).__name__}: {exc}"))
    result.elapsed_seconds = time.time() - started
    store.finish_run(run_id, result.processed, len(result.failed))
    return result


def iter_dataset(root: Path | str | None = None) -> Iterator[Path]:
    """遍历数据集里的合同文件（与 `pipeline.dataset_files` 同口径，供批量用）。"""
    from .pipeline import dataset_files

    for path in dataset_files(root):
        yield Path(path)
