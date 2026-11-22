---
title: 智律契灵 · 合同智能审查
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: 五节点合同审查：要素抽取 + 风险检出 + 主体核查 + 法条匹配 + 综合评分
---

# 智律契灵 · 合同智能审查

上传一份合同（.docx / .pdf / .txt），返回：风险清单（27 个风险码 7 类）、
法律依据（民法典条文 + 引文）、条款原文定位、审查链路 trace、综合评分。

纯 CPU 可跑，单份 3 秒内。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | Web 界面（审查台 / 审查记录 / 系统状态） |
| GET | `/api/health` | 健康检查（PaaS 探活用） |
| POST | `/api/review` | JSON `{"text": "..."}` 或 multipart 文件 |
| POST | `/api/batch` | 批量审查，返回汇总 |
| GET | `/api/history` | 审查记录分页 |
| GET | `/api/stats` | 库统计 |

## 想接大模型

本 Space 默认不加载模型（免费 CPU 档跑不动）。要开节点②的模型分支，
在 Settings → Variables 里填 `ZLQL_LLM_ENDPOINT` / `ZLQL_LLM_MODEL` / `ZLQL_LLM_API_KEY`，
指向任意 OpenAI 兼容服务即可，代码不用改。
