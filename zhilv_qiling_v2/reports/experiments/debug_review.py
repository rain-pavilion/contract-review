import sys
from pathlib import Path

sys.path.insert(0, ".")
from zhilv_qiling_v2.pipeline import review_path  # noqa: E402
from zhilv_qiling_v2.pipeline import build_contract, prepare  # noqa: E402
from zhilv_qiling_v2 import detectors as D  # noqa: E402

path = Path(sys.argv[1])
report = review_path(path)
for risk in report.risks:
    print(risk.code, "| tags:", risk.tags, "| conf", round(risk.confidence, 2), "|", risk.message[:40])

doc = prepare(path.name, path=path)
contract = build_contract(doc)
print("amounts:", [(e.value, e.normalized, e.role) for e in contract.elements.amounts])
print("dates:", [(e.value, e.normalized, e.role) for e in contract.elements.dates])
try:
    print("detect_amounts ->", len(D.detect_amounts(contract)))
    print("detect_dates ->", [(c.code, c.extra) for c in D.detect_dates(contract)])
except Exception as exc:
    import traceback

    traceback.print_exc()
