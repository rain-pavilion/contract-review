import sys
import traceback
from pathlib import Path

sys.path.insert(0, ".")
from zhilv_qiling_v2 import detectors as D  # noqa: E402
from zhilv_qiling_v2.pipeline import build_contract, prepare  # noqa: E402

for name in sys.argv[1:]:
    path = Path(name)
    contract = build_contract(prepare(path.name, path=path))
    print("=" * 60)
    print(path.name, "clauses", len(contract.document.clauses))
    for detector in D.DETECTORS:
        try:
            found = detector(contract)
        except Exception:
            print("  !!", detector.__name__, "raised")
            traceback.print_exc()
            continue
        print(f"  {detector.__name__}: {[(c.code, c.extra.get('issue_code')) for c in found]}")
