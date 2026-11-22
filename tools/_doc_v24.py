"""把 v2.4 的落地内容追加到 README 与 CHANGELOG（一次性脚本）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

README_SECTION = """
## v2.4：申报书剩余缺口的四项落地（2026-11-12）

上一版对照申报书列了九处差异，其中三处是真缺口（节点③"图谱+RAG"名不副实、
"十万份级存储"未压测、"调 LawBench 评估"未做），加上"名字含大模型但链路无 LLM"。
这一版逐项补：

| 差异 | 落地文件 | 实测 |
|---|---|---|
| 节点③ 只有关键词表，没有图谱与依据 | `legal_kg.py` + `data/qualification_kg.jsonl` | 15 条资质条目（13 生效 / 2 已废止）、四类节点三类边 45 节点 30 边；检索首位正确率见 `test_legal_kg.py` 12 例 |
| "十万份级存储"只在 20 份上验过 | `tools/_store_scale.py` | **10 万份写入 177.6 s（563 份/秒）**；库 57.4 MB；深分页 page=5000 75.8 ms；FTS 0.1~17.6 ms；`stats()` 35.5 ms |
| "调 LawBench 评估" 没做 | `lawbench.py` | 六项能力映射到 LawBench 任务类型与指标类型，5 项出数（见下表） |
| 名字含"大模型"但默认链路没有任何模型调用 | `local_llm.py` | 本机 Qwen2.5-1.5B-Instruct（3.10 GB）以 **OpenAI 兼容服务**接入，`ZLQL_LLM_ENDPOINT` 一填即生效；端到端 trace 新增 `logic_llm` 节点 |

### 模型通道怎么接（两种都不改代码）

```bash
# A. 本地小模型（本机已验证：加载 4.5 s，生成 1.17 s / 64 token）
python -m zhilv_qiling_v2.local_llm --serve --port 8620
export ZLQL_LLM_ENDPOINT=http://127.0.0.1:8620/v1
export ZLQL_LLM_MODEL=qwen2.5-1.5b-instruct

# B. 换成申报书里的 LegalOne-8B（vLLM / Ollama / 云网关任一，只要 OpenAI 兼容）
export ZLQL_LLM_ENDPOINT=http://<host>:<port>/v1
export ZLQL_LLM_MODEL=LegalOne-8B
```

配了端点就自动开通道（`ZLQL_ENABLE_LLM=0` 可显式关掉）。**硬护栏不变**：模型给的每条
结论必须能在条款原文里定位到引文，否则丢弃并计入 `rejected`。

### LawBench 式评测结果（`python -m zhilv_qiling_v2.lawbench`）

| LawBench 任务 | 本项目落点 | 指标 |
|---|---|---|
| 2-6 Extraction | 节点① 要素抽取 | micro-F1 **0.7514**（补全金标口径 **0.8812**）、macro 0.7300 |
| 2-3 MLC | 27 风险码检出（对抗集） | P 0.5079 / R **1.0000** / F1 0.6737 |
| 3-1 MLC（排序） | 节点④ 法条推荐 | Recall@5 **0.6944** / @10 0.8889 / MRR@5 **1.0000** |
| 2-4 SLC | 条款—条文一致性（双塔） | Accuracy 0.8750 / macro-F1 0.3994（弱监督） |
| 1-2 SLC | 节点③ 主体资格审查 | 暂无金标，如实报空表 |
| generation | 节点⑤ 复核意见 | 无参考答案，只报可用性 |

**这不是 LawBench 官方分数**：数据是自建 113 份语料 + 对抗集 + 真实政采合同，没有用
LawBench 的划分与标注；对齐的是"任务类型 + 指标类型"，便于横向比较，不冒充榜单成绩。
"""

CHANGELOG_SECTION = """
## 三之五、v2.4：补齐申报书剩余三处缺口（2026-11-12）

**做了什么**

1. `legal_kg.py`（430 行）+ `data/qualification_kg.jsonl`（15 条）：把节点③从"11 组关键词"
   升级为「主体→行业→资质→法规」知识图谱 + BM25 检索，结论必须带 `basis_law` /
   `basis_article` / `source_url`；两条已废止资质（物业服务企业资质、计算机信息系统集成资质）
   显式标 `abolished` 并抑制报警——这类"拿废止资质报缺证"是真实项目里最常见的误报来源。
2. `tools/_store_scale.py`：10 万份压测。写入 177.6 s（563 份/秒、1.78 ms/份）、库 57.4 MB
   （573.5 B/份）、page=5000 深分页 75.8 ms、FTS trigram 0.1~17.6 ms、`stats()` 35.5 ms。
   结论三项全过：十万份可写 / 深分页可接受 / FTS 可用。
3. `local_llm.py`（约 320 行）：本地因果语言模型包成 OpenAI 兼容服务，让 `llm_client` 那条
   链路一行不改就能跑真模型；`<think>` 段按 LegalOne 契约剥离；支持 `--serve` 与 `--prompt`。
4. `lawbench.py`（约 330 行）：六项能力映射 LawBench 任务类型与指标类型，单任务失败不影响
   整表（`_safe`），结果落 `reports/lawbench.json`。
5. `engine.py` 新增 `logic_llm` 节点（节点② 的模型分支真正接进主链）；`config_from_env()`
   支持 `ZLQL_LLM_*` / `ZLQL_USE_BGE` / `ZLQL_NER_MODEL`，且**配了端点自动开通道**——
   "填了地址却没生效"是最难查的一类静默问题。
6. 测试 162 → **175**（新增 `test_local_llm.py` 13 例：strip_think / extract_json 四种形态 /
   describe / LawBench 任务表完整性与单点失败隔离）。

**实测（2026-11-12）**

- 真实本地模型端到端：加载 4.47 s（cuda），5 条条款遍历 19.2 s，**采纳 4 条 / 引文未通过 6 条**，
  `judge_llm` 生成复核意见但**分数仍由规则算（45）**——模型只读，不改分。
- 小模型的真实短板：它把"合同中的某些术语未经核实"套上了 `unverified_property`（该码本义是
  "房产权属未核实"）。**1.5B 只能验证通道可用，达不到可用质量**；要可用得换 7B/8B，
  这也是申报书写 LegalOne-8B 的原因。

**仍未做**

- LegalOne-8B 权重本机取不到（HF 直连不通、ModelScope 无镜像），现用 Qwen2.5-1.5B 验证协议；
- 节点③ 的资质条目只有 15 条，要做成"可查"至少要覆盖常见 30 类行业；
- LawBench 官方划分未使用（需按其格式重训，属下一阶段）。
"""

readme = ROOT / "zhilv_qiling_v2" / "README.md"
text = readme.read_text(encoding="utf-8")
if "v2.4：申报书剩余缺口" not in text:
    readme.write_text(text.rstrip() + "\n" + README_SECTION, encoding="utf-8")
    print("README 追加 v2.4 节")
else:
    print("README 已有 v2.4 节，跳过")

changelog = ROOT / "docs" / "CHANGELOG-v1-v2.md"
text = changelog.read_text(encoding="utf-8")
if "v2.4：补齐申报书剩余三处缺口" not in text:
    changelog.write_text(text.rstrip() + "\n" + CHANGELOG_SECTION, encoding="utf-8")
    print("CHANGELOG 追加 v2.4 节")
else:
    print("CHANGELOG 已有 v2.4 节，跳过")

print("README 行数", len(readme.read_text(encoding='utf-8').splitlines()))
print("CHANGELOG 行数", len(changelog.read_text(encoding='utf-8').splitlines()))
