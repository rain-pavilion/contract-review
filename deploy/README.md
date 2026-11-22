# 云端部署指南

三条路，按"你手上有什么"选。**三者共用同一个 Dockerfile**，不需要改代码。

---

## 现在能直接跑的东西

```
GET  /                 Web 界面（审查台 / 审查记录 / 系统状态）
GET  /api/health       健康检查，PaaS 探活用
POST /api/review       JSON {"text": "..."} 或 multipart 文件
POST /api/batch        批量审查
GET  /api/history      审查记录分页（支持 code / severity 过滤）
GET  /api/history/{id} 单份风险明细
GET  /api/stats        SQLite 库统计
```

本地起：

```powershell
python -m zhilv_qiling_v2.api --port 8611
# 浏览器打开 http://127.0.0.1:8611/
# 想直接看效果：http://127.0.0.1:8611/?demo=1 （自动载入示例并跑一次）
```

自检（不依赖 PowerShell，避免 GBK 解 UTF-8 的坑）：

```powershell
python tools\_web_e2e.py http://127.0.0.1:8611
```

---

## 路一：Hugging Face Spaces（最省事，免费 CPU）

1. 新建 Space → SDK 选 **Docker** → 模板 blank
2. 把仓库内容推上去；`README.md` 用 `deploy/huggingface/README.md` 那份（前面的 front-matter 决定 SDK 与端口）
3. Space 会自动按根目录 `Dockerfile` 构建 —— 但它找根目录，所以要么把 `deploy/Dockerfile` 复制到根，要么在 Space 里加一个一行的根 Dockerfile：`FROM` 不行，得用：
   ```
   # Space 根 Dockerfile
   COPY deploy/Dockerfile /tmp/Dockerfile
   ```
   更简单的做法：**直接把 `deploy/Dockerfile` 的内容放到仓库根的 `Dockerfile`**（内容里路径都是相对根的，不用改）。

构建约 4~6 分钟；冷启动后首页 300 ms 可开。

## 路二：Render（有域名、能跑常驻，free 档会休眠）

`deploy/render.yaml` 已是 Blueprint 格式：

1. 推到 GitHub
2. Render → New → **Blueprint** → 选仓库 → 它读 `deploy/render.yaml`
3. 部署完拿到 `https://<name>.onrender.com`

免费档 15 分钟无请求会休眠，首次唤醒约 30 s。要稳定演示就升到 starter（$7/月）。

## 路三：任意 VPS / 阿里云 ECS（最可控，能挂 GPU）

```bash
git clone <仓库> && cd <仓库>
docker build -f deploy/Dockerfile -t zhilv-qiling .
docker run -d --name zlql -p 80:8000 --restart unless-stopped zhilv-qiling
```

要 HTTPS 就在前面挂 Nginx 或 Caddy（Caddy 两行配置自动签证书）。

---

## 你问的 32 GB —— 是内存，不是显存

实测这台机器：

| 项 | 数值 | 来源 |
|---|---|---|
| 显卡 | NVIDIA GeForce RTX 5070 **Laptop** GPU | `nvidia-smi` |
| 显存 | **8151 MiB**（空闲 5484 MiB） | `nvidia-smi` |
| 系统内存 | **31.37 GB** | `Win32_ComputerSystem.TotalPhysicalMemory` |
| 驱动 / CUDA | 610.74 / torch 2.9.1+cu128 / sm_120 | `torch.cuda.get_device_properties` |

RTX 5070 **Laptop** 官方就是 8 GB GDDR7；32 GB 是插在主板上的内存条。所以"本地跑 8B 模型"这条路在当前机器上不成立（8B fp16 约 16 GB 显存，4bit 量化约 5.5 GB，勉强能塞但会和系统抢显存）。

### 那 8B 怎么跑？走云端

LegalOne-8B 权重 16.42 GB，两种可行的云端接法：

**A. 云 GPU 起 vLLM（推荐，一次配好长期用）**

```bash
# 在 A10 / L20 / 4090 等 ≥24 GB 显存的实例上
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model CSHaitao/LegalOne-8B \
  --served-model-name legalone-8b \
  --max-model-len 8192 --port 8000
```

然后把 Web 服务指过去（代码零改动）：

```
ZLQL_LLM_ENDPOINT=https://<你的实例>/v1
ZLQL_LLM_MODEL=legalone-8b
ZLQL_LLM_API_KEY=<如果网关要>
ZLQL_LLM_MAX_CLAUSES=24
```

**B. 现成 API**（DeepSeek / 通义 / 智谱都兼容 OpenAI 协议）：把 endpoint 换成它们的 `/v1` 即可。

接上之后，节点②的模型分支会真正产出候选，trace 里会多出"模型遍历 N 条条款 / 采纳 X / 引文未通过 Y"，而**分数仍由规则算**（这是护栏：模型只读不写）。

---

## 部署后自查清单

```powershell
# 1. 首页 200
curl https://<域名>/
# 2. 健康检查
curl https://<域名>/api/health
# 3. 跑一份真实合同
python tools\_web_e2e.py https://<域名>
```

三项都过就说明镜像没问题。若 `/api/review` 报 500，先看日志里有没有
`FileNotFoundError: data/法条结构化数据.jsonl` —— 那是 Dockerfile 的 COPY 路径写错了。
