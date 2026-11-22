"""临时评测：对抗样本集 issue_codes 金标准 vs v2 检测器输出（码表对齐后）。"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from zhilv_qiling_v2.pipeline import review_path  # noqa: E402

ROOT = Path("data/generated_contracts/samr_multi_template_2025/adversarial")
manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))

tp = fp = fn = 0
rows = []
per_code = {}
started = time.time()
for record in manifest["records"]:
    path = ROOT / record["docx"]
    if not path.exists():
        continue
    report = review_path(path)
    got_codes = {r.code for r in report.risks if r.code != "no_substantive_risk"}
    got_issue = {t for r in report.risks for t in r.tags}
    gold = set(record.get("issue_codes") or [])
    # 主评测：数据集缺陷码层面（tags 对齐）
    inter = got_issue & gold
    tp += len(inter)
    fp += len(got_issue - gold)
    fn += len(gold - got_issue)
    for code in gold:
        stat = per_code.setdefault(code, {"tp": 0, "fn": 0, "fp": 0})
        stat["tp" if code in inter else "fn"] += 1
    for code in got_issue - gold:
        per_code.setdefault(code, {"tp": 0, "fn": 0, "fp": 0})["fp"] += 1
    rows.append((record["contract_id"], sorted(gold), sorted(got_issue), sorted(got_codes), report.score))

elapsed = time.time() - started
prec = tp / (tp + fp) if tp + fp else 0.0
rec = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
print(f"contracts={len(rows)} elapsed={elapsed:.1f}s avg={elapsed/max(1,len(rows)):.2f}s")
print(f"issue-code level TP={tp} FP={fp} FN={fn} P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
hit = sum(1 for r in rows if set(r[1]) & set(r[2]))
print(f"contract-level hit {hit}/{len(rows)}")
print("per-code:")
for code, stat in sorted(per_code.items()):
    print(f"  {code:26s} tp={stat['tp']} fn={stat['fn']} fp={stat['fp']}")
print("rows:")
for contract_id, gold, got_issue, got_codes, score in rows:
    print("  ", contract_id, "gold=", gold, "hit=", got_issue, "score=", score)
