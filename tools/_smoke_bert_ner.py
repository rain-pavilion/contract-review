"""BERT-CRF 冒烟测试：只用少量文档跑 1 个 epoch，确认 CUDA/transformers API 都对。

真正的训练走 `python -m zhilv_qiling_v2.reports.experiments.run_gpu_upgrade ner`，
这个脚本只负责在跑长任务之前把接口问题暴露出来（省得等 20 分钟才报错）。
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from zhilv_qiling_v2 import bert_ner, config, pretrained, train


def main() -> int:
    print("[1] 权重目录:", pretrained.resolve())
    print("[2] 设备:", bert_ner.device_of())
    docs = train.collect_docs()
    gold = [doc for doc in docs if doc.gold_entities]
    print(f"[3] 全量 {len(docs)} 份，带金标 {len(gold)} 份")
    if not gold:
        print("[x] 没有带金标的文档")
        return 1

    subset = gold[:4]
    began = time.time()
    result = bert_ner.train(
        subset,
        config.default_config(),
        epochs=1,
        batch_size=2,
        limit=2,
        use_silver=True,
    )
    print(f"[4] 训练完成，用时 {time.time() - began:.1f}s")
    print("[5] 训练回执:", {k: result[k] for k in ("windows", "epochs", "device")})
    valid = result["valid"]
    print("[6] 留出集 macro_f1:", valid["macro_f1"], "逐类 F1:",
          {kind: row[2] for kind, row in valid["per_kind"].items()})

    model, tokenizer, _meta = bert_ner.load_model()
    spans = bert_ner.predict(
        model,
        tokenizer,
        "甲方：星海科技产业园管理有限公司；合同金额：人民币1200000元，2026年3月1日签订。",
    )
    print("[7] 推理样例:", spans[:6])
    return 0


if __name__ == "__main__":
    sys.exit(main())
