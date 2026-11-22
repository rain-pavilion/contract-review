"""把 clause_fts 换成 trigram 分词（中文子串检索），并修 search_clauses 的查询构造。

问题：默认 `unicode61` 分词器把整句中文当成一个 token，"违约责任"永远匹配不上。
SQLite 3.34+ 的 `trigram` 分词器支持中文子串匹配（3 字起）。
"""

import pathlib

p = pathlib.Path("zhilv_qiling_v2/store.py")
text = p.read_text(encoding="utf-8")

old_schema = '''_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS clause_fts USING fts5(
    contract_id UNINDEXED,
    clause_id   UNINDEXED,
    body
);
"""'''
new_schema = '''#: 中文检索必须用 trigram 分词器：默认 unicode61 会把整句当成一个 token，
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
"""'''
assert old_schema in text
text = text.replace(old_schema, new_schema, 1)

old_init = '''        self.fts = True
        try:
            self._conn.executescript(_FTS_SCHEMA)
        except sqlite3.OperationalError:
            self.fts = False'''
new_init = '''        # 优先 trigram（中文子串可检索）；不支持则退回普通 fts5；再不行就只用 LIKE
        self.fts = "trigram"
        for candidate, name in ((_FTS_TRIGRAM, "trigram"), (_FTS_PLAIN, "plain")):
            try:
                self._conn.executescript(candidate)
                self.fts = name
                break
            except sqlite3.OperationalError:
                continue
        else:
            self.fts = ""'''
assert old_init in text
text = text.replace(old_init, new_init, 1)

old_save = '''        if not self.fts:
            return 0'''
new_save = '''        if not self.fts:
            return 0'''
assert old_save in text

old_search = '''    def search_clauses(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """按条款正文找合同；FTS 不可用时退化为 LIKE。"""
        if self.fts:
            try:
                rows = self._conn.execute(
                    """SELECT contract_id, clause_id, snippet(clause_fts, 2, '[', ']', '…', 12) AS snippet
                       FROM clause_fts WHERE clause_fts MATCH ? LIMIT ?""",
                    (_fts_query(query), limit),
                ).fetchall()
                return [dict(row) for row in rows]
            except sqlite3.OperationalError:
                pass'''
new_search = '''    def search_clauses(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
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
                return [dict(row) for row in rows]'''
assert old_search in text
text = text.replace(old_search, new_search, 1)

old_fts_query = '''def _fts_query(text: str) -> str:
    """把用户输入转成 FTS5 查询：每个汉字片段加引号，避免被当成语法。"""
    tokens = [token for token in str(text).replace('"', " ").split() if token]
    if not tokens:
        return '""'
    return " OR ".join(f'"{token}"' for token in tokens)'''
new_fts_query = '''def _fts_query(text: str) -> str:
    """trigram 分词器下，整串加引号就是子串检索；多段用 OR 连。"""
    tokens = [token for token in str(text).replace('"', " ").split() if token]
    if not tokens:
        return '""'
    return " OR ".join(f'"{token}"' for token in tokens)'''
assert old_fts_query in text
text = text.replace(old_fts_query, new_fts_query, 1)

p.write_text(text, encoding="utf-8")
print("store.py FTS 已换 trigram")
