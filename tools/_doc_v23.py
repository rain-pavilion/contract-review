"""把 v2.3（存储层 + 统一评测台）写进 README 与 CHANGELOG。"""

from __future__ import annotations

import pathlib

README = pathlib.Path("zhilv_qiling_v2/README.md")
text = README.read_text(encoding="utf-8")

section = """
## 存储与统一评测台（v2.3）

申报书指标里"支持十万份级存储 / 检索 / 批量审查调度"与"系统化评估"此前是空白：
v1 用 qdrant 做向量检索，但审查结果本身没有归档层，评测也只有零散实验脚本。
v2.3 补上这两块，都是纯标准库，笔记本可直接跑。

### `store.py`：SQLite 审查结果库

| 表 | 内容 | 用途 |
|---|---|---|
| `contracts` | 来源、类型、评分、风险数、耗时、文本指纹 | 列表页与统计 |
| `risks` | 风险码、分类、等级、置信、是否待复核、标签 | 按码/按级筛选 |
| `citations` | 法条名、条号、`article_uid` | 引用溯源（只存索引里真实存在的条） |
| `runs` | 批量任务起止、处理数、失败数 | 调度账本 |
| `clause_fts` | 条款正文（FTS5 `trigram` 分词） | "哪份合同写过这句话" |

三个设计取舍：

1. **单文件 SQLite**：零依赖，`PRAGMA journal_mode=WAL` + 三个索引，百万行级分页仍是毫秒。
2. **断点续跑**：`batch_to_store(..., resume=True)` 按 `contract_id` 主键跳过已完成项，
   重跑只补新文件（实测第二次 `processed=0 skipped=20`）。
3. **中文全文检索必须用 trigram**：默认 `unicode61` 把整句当一个 token，
   "违约责任"永远匹配不上（实测返回 0 条）；换 `trigram` 后子串可检索，不可用时自动退化 LIKE。

### `bench.py`：三层统一评测台

| 任务 | 指标 | 口径来源 |
|---|---|---|
| `retrieval` | 强/弱金标准分层 R@1..10、MRR@5 | **复用 `train.evaluate_retrieval`**（与 v1 脚本同协议） |
| `detection` | 逐码 TP/FP/FN、合同级命中 | 对抗集 `issue_codes` 为金标准 |
| `latency` | P50/P95/最大、份每分钟 | 固定样本集端到端 |

指标只允许一处实现：bench 第一版自己重算过一遍，得出 rrf 0.875 / ltr 0.125 这种与论文
（0.6278 / 0.6944）冲突的数字 —— 同一指标两个口径是答辩硬伤，已改为直接调用训练脚本里的那套。

### 实测（2026-11-11，`python -m zhilv_qiling_v2.cli bench`）

```
检索 rrf_only     strong R@5=0.6278 R@10=0.6556 MRR@5=1.0000
检索 rrf_plus_ltr strong R@5=0.6944 R@10=0.8889 MRR@5=1.0000
检出 P=0.508 R=1.000 F1=0.674 合同 10/10
延迟 P50=1.52s P95=3.09s 首份=0.38s 吞吐=49 份/分
```

批量入库实测：20 份对抗合同 → 151 条风险 / 293 条引用，评分区间 0~51、均值 11.95，
平均 468 ms/份；`store stats` 与 `store list` 分页、`store search 押金` 条款检索均正常。

### 命令行

```bash
python -m zhilv_qiling_v2.cli review samples/tech-development-contract.txt --store   # 审查并入库
python -m zhilv_qiling_v2.cli batch --root data/generated_contracts --limit 20 --store
python -m zhilv_qiling_v2.cli store stats
python -m zhilv_qiling_v2.cli store list --size 10 --severity high
python -m zhilv_qiling_v2.cli store risks ct_xxxxxxxxxx
python -m zhilv_qiling_v2.cli store search 押金
python -m zhilv_qiling_v2.cli bench --tasks retrieval,detection,latency
```

### 同批修掉的两个"静默漂移"

1. **BGE-M3 改为 opt-in**：接上 BGE 当密集通道后强金标准 R@5 从 0.628 掉到 0.167，
   但代码原是无条件启用，导致之后所有评测都在 BGE 上跑、复现不出论文数字。
   现在 `Config.use_bge=False`（`ZLQL_USE_BGE=1` 显式开启）。
2. **`rerank_ltr.json` 补 `normalize` / `fusion_weights` / `rrf_k`**：
   训练脚本手写 JSON 时漏了这三个键，推理端读不到 `normalize` 就退回逐条打分（量纲差两个
   数量级），+LTR 的 R@5 从 0.6944 掉到 0.2222。补上后 `bench` 与论文数字一致。

单元测试：**153 → 162**（新增 `tests_v2/test_store.py` 9 例：落库读回、幂等、分页过滤、
中文子串检索、统计一致性、断点续跑、评测台形状与可序列化）。
"""

anchor = "## 降级矩阵"
if anchor in text:
    text = text.replace(anchor, section.strip() + "\n\n" + anchor, 1)
else:
    text = text.rstrip() + "\n\n" + section.strip() + "\n"
README.write_text(text, encoding="utf-8")
print("README 已追加 v2.3 节")

CHANGELOG = pathlib.Path("docs/CHANGELOG-v1-v2.md")
change = CHANGELOG.read_text(encoding="utf-8")
entry = """
## 三之四、v2.3：存储层与统一评测台（2026-11-11）

**为什么补**：申报书写的"十万份级存储/检索/批量审查调度"和"系统化评估"，在 v2.2 之前是空白 ——
v1 的 qdrant 是向量检索，不是审查结果归档；评测散在若干实验脚本里，没有统一协议。

**做了什么**

- `store.py`：SQLite 单文件库（`contracts` / `risks` / `citations` / `runs` / `clause_fts`），
  WAL + 三索引，分页、按码/按级过滤、断点续跑、条款全文检索（FTS5 `trigram`，中文子串必需）。
- `bench.py`：三层基准（检索 / 检出 / 延迟），检索指标**复用** `train.evaluate_retrieval`，
  结果落 `reports/bench_*.json`。
- CLI：`store stats|list|risks|search`、`bench`，以及 `review/batch/explain --store`。
- 测试：153 → 162（`test_store.py` 9 例）。

**同批修的两个静默漂移**

1. BGE-M3 从"无条件当密集通道"改为 opt-in（`use_bge=False`）：它是负结果（R@5 0.628→0.167），
   默认开着会让后续所有评测复现不出论文数字。
2. `rerank_ltr.json` 补 `normalize`/`fusion_weights`/`rrf_k`：缺失时推理端退回逐条打分，
   +LTR 的 R@5 从 0.6944 掉到 0.2222。

**实测（可复现）**

| 项目 | 数字 |
|---|---|
| 检索 rrf_only strong | R@5 0.6278 / R@10 0.6556 / MRR@5 1.0000 |
| 检索 rrf_plus_ltr strong | R@5 0.6944 / R@10 0.8889 / MRR@5 1.0000 |
| 检出（对抗集） | P 0.508 / R 1.000 / F1 0.674，合同级 10/10 |
| 延迟 | P50 1.52 s / P95 3.09 s / 首份 0.38 s / 49 份每分 |
| 批量入库 | 20 份 → 151 风险 / 293 引用，均值 468 ms/份 |

**仍未做**：真实 LawBench / PROVDATA 数据集不可得（需申请或需 GPU 微调），
`bench` 的检索任务只对齐其**指标形态**，不冒充数据集复现。
"""
CHANGELOG.write_text(change.rstrip() + "\n" + entry, encoding="utf-8")
print("CHANGELOG 已追加 v2.3")
