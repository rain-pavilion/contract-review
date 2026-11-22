import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from zhilv_qiling_v2 import text as T  # noqa: E402
from zhilv_qiling_v2.parsers import parse_path  # noqa: E402

ROOT = Path("data/generated_contracts/samr_multi_template_2025/adversarial")
manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
out = []
for record in manifest["records"]:
    path = ROOT / record["docx"]
    if not path.exists():
        continue
    parsed = parse_path(path)
    body = "\n".join(b.text for b in parsed.blocks)
    out.append("=" * 70)
    out.append(f"{record['contract_id']}  gold={record['issue_codes']}")
    out.append(body)
target = Path("zhilv_qiling_v2/reports/_adv_texts.txt")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text("\n".join(out), encoding="utf-8")
print("written", len(out))
print(T.full_to_half("Ａ１"))
