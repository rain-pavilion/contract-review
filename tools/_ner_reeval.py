"""用已训练好的 BERT-CRF 权重重算评测指标（不重训）。

动机：`bert_ner.predict()` 后来加了两道金额后处理闸（大写重复 / 单价），
但 `reports/gpu_upgrade_bert_ner.json` 里还是加闸之前的数字。
重训要 19 分钟，而口径变化只影响推理，所以这里只走「加载 → 预测 → 统计」。

用法：
    python tools/_ner_reeval.py            # 打印并写回报告
    python tools/_ner_reeval.py --dry-run  # 只打印
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhilv_qiling_v2 import bert_ner as B  # noqa: E402
from zhilv_qiling_v2 import train as T  # noqa: E402

REPORT = ROOT / "zhilv_qiling_v2" / "reports" / "gpu_upgrade_bert_ner.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写回报告")
    args = ap.parse_args()

    if not B.available():
        print("BERT-CRF 权重或 torch 不可用", file=sys.stderr)
        return 2

    docs = T.collect_docs()
    _, valid = T.split_docs(docs, ratio=0.25)
    model, tokenizer, meta = B.load_model()[:3]

    res = B.evaluate(model, tokenizer, valid)
    stats = res
    micro = res["micro"]
    complete = res["micro_complete"]
    print(f"留出集 {len(valid)} 份 ｜ 计入 {res.get('docs')} 份")
    print(
        "严格口径 micro  P{:.4f} R{:.4f} F1 {:.4f}  (tp{} fp{} fn{})".format(
            micro["precision"], micro["recall"], micro["f1"],
            micro["tp"], micro["fp"], micro["fn"],
        )
    )
    print(
        "补全口径 micro  P{:.4f} R{:.4f} F1 {:.4f}  (tp{} fp{} fn{})".format(
            complete["precision"], complete["recall"], complete["f1"],
            complete["tp"], complete["fp"], complete["fn"],
        )
    )
    print(f"宏平均 F1       {stats['macro_f1']:.4f}")
    for kind, row in sorted(stats["per_kind"].items()):
        print("  {:<8} P{:.4f} R{:.4f} F1 {:.4f}".format(kind, row[0], row[1], row[2]))
    if stats.get("party_loose"):
        loose = stats["party_loose"]
        print(
            "PARTY 宽松      P{:.4f} R{:.4f} F1 {:.4f}".format(
                loose["precision"], loose["recall"], loose["f1"],
            )
        )

    if args.dry_run:
        return 0

    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    valid_block = payload.setdefault("new", {}).setdefault("valid", {})
    if "micro" in valid_block and "micro_before_gates" not in valid_block:
        valid_block["micro_before_gates"] = valid_block["micro"]
    valid_block["micro"] = micro
    valid_block["micro_complete"] = complete
    valid_block["macro_f1"] = stats["macro_f1"]
    valid_block["per_kind"] = stats["per_kind"]
    if stats.get("party_loose"):
        valid_block["party_loose"] = stats["party_loose"]
    payload.setdefault("postprocess", {})
    payload["postprocess"] = {
        "amount_duplicate_capital": "实体为中文大写且 16 字内有阿拉伯金额 → 视为重复，丢弃",
        "amount_unit_price": "实体前 14 字内有单价/税率/每平方米等线索且自身非总价 → 丢弃",
        "evaluated_by": "tools/_ner_reeval.py（不重训，只重算推理口径）",
        "model_meta": meta if isinstance(meta, dict) else {},
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"已写回 {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
