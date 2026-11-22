"""给 README 与 CHANGELOG 追加 v2.5 节：主脑调度 + 分级复核。

为什么单独写脚本而不是手改：README 含大量中文与代码块，手工编辑容易把
围栏或表格打断；这里按"文件末尾追加"的方式写，最小化冲突面。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "zhilv_qiling_v2" / "README.md"
CHANGELOG = ROOT / "docs" / "CHANGELOG-v1-v2.md"

README_SECTION = """

## 主脑调度与分级复核（v2.5）

申报书里"各节点将子任务派发给专门的大模型或工具"这句，此前只是把五个函数按顺序调一遍。
v2.5 补了两个真实 Agent，并把调用改成**分级**——昂贵的 8B 只碰最该碰的条款。

### 图结构（拓扑即白名单）

```
intake ──→ orchestrator ──→ ner / logic / subject / statute
              ↑                    │
              │  defects.detected  │
              └───── logic ←────────┘
orchestrator ──→ reviewer ──→ judge
```

- `MessageBus(strict=True)` 只放行 `TOPOLOGY` 里声明过的边，越界投递直接拒收并计数；
- 轨迹可导出（`MultiAgentRun.dump_trace()`），"谁在什么时候给谁发了什么"落在 JSON 里。

### 主脑（orchestrator）跑两趟

| 轮次 | 触发 | 产出 |
| --- | --- | --- |
| 第一趟 | `intake` 交来解析结果 | 给四个功能节点发 `task.assign`，带**档位**（`rules` / `llm` / `local`）与模型名 |
| 第二趟 | `logic` 回报 `defects.detected` | 按「严重度 × 100 + 置信度」排序取前 `llm_review_top_k` 条条款，发 `review.assign` 给 `reviewer` |

没配 `ZLQL_LLM_STRONG_ENDPOINT` 时第二趟直接跳过并记一条 note——不会让每条合同都去等一个不存在的服务。

### 分级复核（reviewer）

`reviewer` 只跑主脑指派的那几条条款，配置独立：

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `ZLQL_LLM_STRONG_ENDPOINT` | 空 | 强模型端点（如 4bit 的 LegalOne-8B）；空则关闭分级复核 |
| `ZLQL_LLM_STRONG_MODEL` | `legalone-8b` | 强模型名 |
| `ZLQL_LLM_REVIEW_TOP_K` | `3` | 只复核风险最高的前 K 条条款 |
| `ZLQL_LLM_STRONG_TIMEOUT` | `600` | 强模型超时（8B 4bit 在本机约 2.25 token/s，90 s 一定不够） |

### 实测（`samples/tech-development-contract.txt`，24 条款）

| 档位 | 耗时 | 评分 | 风险 | 说明 |
| --- | --- | --- | --- | --- |
| 规则全量 | 0.55 s | 83 | 3 | 无模型 |
| 1.7B 全量（6 条款） | 49.0 s | 58 | 6 | 3 条模型产出，误套 `unverified_property` |
| 8B 全量（6 条款） | 921.5 s | 59 | 7 | 4 条模型产出 |
| **规则全量 + 8B 复核 top-1** | **99.8 s** | **77** | **4** | 8B 补出 `jurisdiction_ambiguous`；比全量快 **9.2 倍** |

### 插件位

外部模块可以往图里加节点：

```python
from zhilv_qiling_v2 import agents_multi as M

def handle(ctx):
    ctx.emit("auditor", "judge", "audit.done", {"ok": True})
    return {"notes": ["插件节点跑过了"]}

M.register_agent(
    "auditor",
    {"role": "合规复核", "executor": "custom", "model": "plugin"},
    handle,
    upstream=("statute",),
    downstream=("judge",),
)
```

注册后 `agent_specs()` 与 `TOPOLOGY` 同步更新，`build_agents()` 会自动带上它；
拓扑外投递照样拒收，所以插件也必须守图。

### 测试

`zhilv_qiling_v2/tests_v2/test_agents_dispatch.py`（5 例）守住三件事：主脑开工前确实派活给四个节点、
没配强模型端点时不派复核、插件注册后拓扑与调度同步。全量 **206 tests OK**。
"""

CHANGELOG_SECTION = """

### 三之六、v2.5：主脑调度与分级复核（2026-11-13）

**为什么补**：申报书写"各节点将子任务派发给专门的大模型或工具"，实际是单进程顺序调用五个函数；
且 8B 全量跑一份合同要 921 秒，不可能按"每份都上 8B"落地。

**做了什么**：
1. `agents_multi.py` 加 `orchestrator`（主脑）与 `reviewer`（强模型复核）两个 Agent，拓扑重写为
   `intake → orchestrator → 四节点`，`logic → orchestrator` 作为回报边，`reviewer → judge`；
2. 主脑跑两趟：先按合同规模派活（带档位），再按 logic 回报的缺陷挑 top-K 派强模型复核；
3. `nodes.node_logic` 增 `clauses=` 关键字——只对指定条款跑模型分支，规则与检测器仍走全文；
4. 四个新配置：`llm_strong_endpoint` / `llm_strong_model` / `llm_review_top_k` / `llm_strong_timeout`
   （对应 `ZLQL_LLM_STRONG_*`、`ZLQL_LLM_REVIEW_TOP_K`）；
5. `api.py` 的 `/api/health` 增 `llm.strong` 段；`webui.html` 系统页展示分级复核是否开启、强模型名与 top-K；
6. 插件位 `register_agent(name, spec, handler, upstream, downstream)`。

**实测**：规则全量 0.55 s / 83 分 / 3 风险 → 规则全量 + 8B 复核 top-1 **99.8 s / 77 分 / 4 风险**，
比 8B 全量（921 s）快 9.2 倍，且多检出一条 `jurisdiction_ambiguous`。

**测试**：新增 `test_agents_dispatch.py`（5 例），全量 201 → **206**。
"""


def _append(path: pathlib.Path, text: str) -> None:
    body = path.read_text(encoding="utf-8")
    if "v2.5" in body and "主脑" in body:
        print("SKIP:", path.name, "已有 v2.5 节")
        return
    path.write_text(body.rstrip() + "\n" + text, encoding="utf-8")
    print("OK  :", path.name, "→", len(path.read_text(encoding="utf-8")), "字符")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    _append(README, README_SECTION)
    _append(CHANGELOG, CHANGELOG_SECTION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
