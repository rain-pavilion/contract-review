import sys
from pathlib import Path

sys.path.insert(0, ".")
from zhilv_qiling_v2.parsers import parse_path  # noqa: E402
from zhilv_qiling_v2.pipeline import prepare  # noqa: E402

path = Path(sys.argv[1])
parsed = parse_path(path)
print("blocks", len(parsed.blocks))
for block in parsed.blocks:
    print(block.kind, "|", block.text[:130].replace("\n", " "))
doc = prepare(path.name, path=path)
print("clauses", len(doc.clauses))
for clause in doc.clauses:
    print(clause.clause_id, clause.level, clause.label, "|", (clause.body or "")[:100].replace("\n", " "))
