"""诊断条款全文索引为什么查不到。"""

import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")

conn = sqlite3.connect("data/reviews.sqlite")
conn.row_factory = sqlite3.Row

print("表:", [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")])
try:
    print("fts 行数:", conn.execute("SELECT COUNT(*) FROM clause_fts").fetchone()[0])
    row = conn.execute("SELECT contract_id, clause_id, substr(body,1,60) b FROM clause_fts LIMIT 2").fetchall()
    for item in row:
        print("样例:", dict(item))
    print("trigram 匹配:", conn.execute(
        "SELECT COUNT(*) FROM clause_fts WHERE clause_fts MATCH ?", ('"违约责任"',)).fetchone()[0])
    print("LIKE 匹配:", conn.execute(
        "SELECT COUNT(*) FROM clause_fts WHERE body LIKE ?", ("%违约责任%",)).fetchone()[0])
except sqlite3.OperationalError as exc:
    print("fts 查询失败:", exc)

print("合同数:", conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0])
conn.close()
