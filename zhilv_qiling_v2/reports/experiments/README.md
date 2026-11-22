# 实验与验证脚本（v2 自己的）

这些脚本原本临时写在原项目的 `scripts/` 目录里做验证，为了**不污染原项目**已经全部
移到本目录（原 `scripts/` 目录里 v1 的 10 个脚本一个都没动）。

所有脚本都按"工作目录 = 仓库根（`C:\Users\ROG\OneDrive\Desktop\LLM`）"编写，
并且只读取 `data/` 与 `zhilv_qiling_v2/`，不写原项目目录。

| 脚本 | 用途 | 典型输出 |
|------|------|----------|
| `corpus_stats.py` | 统计 `data/gov_contracts` 的页数/扫描比例/字段完整度 | 63 份 / 1265 页 / 扫描页 1152（91.1%） |
| `check_split.py` | 解析→版面→切分四段串联，检查条款树与覆盖 | 样本 21 条款、0 issue |
| `check_ner.py` | 要素抽取六类 + 一致性信号 | 当事人/金额/日期/标的/编号/期限 + 信号 |
| `check_retrieve.py` | 三路召回 + 加权 RRF + 重排的位次观察 | 单查询 9.5~15.6 ms |
| `ablate_retrieval.py` | 逐通道消融（AST 静态摘取 v1 的 `ISSUE_SPECS`，不 import v1） | 各通道 Recall@k / MRR@5 |
| `eval_adversarial.py` | 对抗集按 `issue_codes` 计 TP/FP/FN/P/R/F1 | 32/32 缺陷、合同级 10/10、F1 0.727 |
| `print_env.py` | 打印包版本与能力探测 | pymupdf/sklearn/requests/ocr 各 True |
| `api_map.py` | AST 提取原项目 API 地图（只读，用于比对 v1→v2 接口差异） | 模块/类/函数签名 |
| `dump_docx_text.py` / `dump_doc_text.py` | 导出 docx/老式 .doc 正文（写临时目录） | 段落 + 表格文本 |
| `dump_adv_texts.py` | 把对抗集全文与金标落盘，调词表时对照 | `reports/_adv_texts.txt` |
| `dump_bib.py` | 把 `docs/refs/refs.bib` 摘成"作者/标题/venue/年/卷/期/页/DOI"清单 | 34 条，供论文参考文献逐条核对 |
| `debug_review.py` / `debug_detectors.py` | 单个样本的中间态调试（候选/标签/检测器输出） | 定位"检测器有输出但最终丢失" |
