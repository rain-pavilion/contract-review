# LegalOne 接入与调试记录

本文记录"把 LegalOne 真正接进节点②"的完整过程：权重怎么下、镜像怎么抖、模型怎么胡说、
哪些坑是模型的问题、哪些是我自己的代码问题。所有数字都是本机实测。

---

## 一、为什么选 1.7B 而不是申报书写的 8B

申报书写的是 LegalOne-8B（16.42 GB，4 个 safetensors 分片）。本机是 RTX 5070 Laptop，
**显存 8151 MiB，空闲约 5081 MiB**。8B 的 bfloat16 权重需要约 16.4 GB 显存，
即便 4bit 量化也要 5 GB 左右且需要 bitsandbytes 的 CUDA 轮子，风险高、耗时长。

所以先接 **LegalOne-1.7B（4.06 GB 单文件）**：架构与 8B 同源（`Qwen3ForCausalLM`），
fp16/bfloat16 在本机可整块装入。**接口与 8B 完全一致**——换权重只改 `--model-dir`，
不改一行调用代码。8B 作为后续可选项，记录在第 5 节。

| 版本 | 体积 | 分片 | 本机可行性 |
|---|---|---|---|
| LegalOne-8B | 16.42 GB | 4 片（4.119 / 4.907 / 4.966 / 2.399 GB） | 需 4bit 或 CPU offload |
| LegalOne-1.7B | 4.06 GB | 单文件 `model.safetensors` | 已实测可跑 |

## 二、下载：镜像抖动不是封锁

`huggingface.co` 在本机 **ConnectionError**（与 github 同）。`hf-mirror.com` 可用，但**极不稳定**：

```
同一条 URL 连续两次请求：503 → 206
小文件 config.json：连续 503 十余次；tokenizer.json 却 1.7 s 拿到
大文件 model.safetensors：6 线程 16 MB 块只抢到 8 块（共 242 块）
```

**先前的判断错在哪**：我只查了 ModelScope 的 `api/v1/models/CSHaitao/LegalOne-8B`，
返回 404 就断言"两个镜像都没有"。实际 hf-mirror 的仓库 API 与 resolve 都是 200：

```
hf-mirror.com/api/models/CSHaitao/LegalOne-8B              → 200 / 5952 B
hf-mirror.com/.../resolve/main/config.json                 → 200 / 1538 B
hf-mirror.com/.../model.safetensors  HEAD                  → 200  CL=4063515640
hf-mirror.com/.../model.safetensors  Range 0-16M           → 206  bytes 0-16777215/4063515640
```

**结论：ReadTimeout / 503 是抖动，加重试就能过；不是封锁。** 这与之前对 torch wheel 的判断一致
（单连接会被掐，分块 + 重试就能下完）。

另一个坑：**镜像对大文件会 302 到 `https://cdn-lfs-us-1.hf.co/...`，直连返回 403**
（签名域名不可直达）。所以不能用"跟随重定向拿直链"的办法，只能靠镜像自身的 Range 支持。

最终成功的参数：

```
python tools\_dl_hf.py --repo CSHaitao/LegalOne-1.7B --dest zhilv_qiling_v2/models/pretrained/legalone-1.7b \
  --threads 2 --chunk-mb 48 --delay 0.8
→ 81 块 / 2077.5 s / 4.06 GB（前段 8.9 MB/s，后段降到 2 MB/s）
```

配套的四处加固（都在 `tools/_dl_hf.py`）：请求超时 30 → 60 s；`total_size` 包 3 次重试
（HEAD 拿不到就用 `Range: bytes=0-0` 读 `Content-Range`）；**小文件（≤1 MB）不走探大小，
直接整份 GET + 8 次重试**（探大小这一步本身也会撞 503）；小文件 4 路并发、大文件串行。

`config.json` / `tokenizer.json` / `tokenizer_config.json` / `vocab.json` / `merges.txt` /
`generation_config.json` 从 **ModelScope 的 `Qwen/Qwen3-1.7B-Base`**（同一架构）秒级拿到。
实测 config：`architectures=['Qwen3ForCausalLM']`、28 层、hidden 2048、`torch_dtype=bfloat16`。

## 三、十二个问题：哪些是模型，哪些是我的

按发现顺序。**前 11 个都是我的代码或配置问题**，只有最后一个是模型能力问题。

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | `TypeError: prepare() got multiple values for 'source_name'` | `prepare` 真实签名是 `prepare(source_name, content=None, *, path=None, config=None)`，返回 `DocumentParse` 而不是 `Contract` | 改 `prepare(path.name, path.read_bytes())`，条款取 `doc.clauses` |
| 2 | 提示词里"条款编号："空着 | 真实合同的 `clause.label` 是**空字符串** | 兜底 `label = clause.label.strip() or f"第{order}条"` |
| 3 | `KeyError: slice(None, 3, None)` | `defects` 是 **dict** 不是 list | 新增 `normalize_payload()` 归一 |
| 4 | 输出尾部无限重复 `<|im_end|>`（decode 成"舅"） | `eos_token_id` 只有 `<|endoftext|>`，模型不停 | `_stop_ids()` 收齐 `<|im_end|>`/`<|endoftext|>` 传给 `generate`，`_decode()` 在首个停止符截断 |
| 5 | 单条款 72 s | 同上（在不停生成重复符） | 修完 **5.7 s** |
| 6 | 输出变 `）冻得到取出）冻存取出）…` 循环乱码 | `no_repeat_ngram_size=8` 有害 | 去掉，只留 `repetition_penalty=1.05` |
| 7 | `{"defects":[{"clause_id":", "code":", ...}]}` 键名齐、值全空 | 我把 `schema_hint="defects:array"` 拼进了 system，模型当成"请列出字段名" | **`schema_hint` 只作调用方自文档，不拼进提示词** |
| 8 | 单条 44~70 s、越写越长 | `max_tokens` 默认 1500 | `llm_max_tokens` 默认 **512** |
| 9 | `<think>` 占 796/1004 字，JSON 被埋在推理里 | LegalOne 的 `<think>…</think>` 契约 | `strip_think()` 先剥再解析；未闭合单独报错 |
| 10 | 各种 JSON 走形（裸键名、引号错位、尾逗号、缺成员逗号、括号不闭合） | 小模型生成不稳定 | `_repair_json()` 六步修复链，自测 6/6 |
| 11 | `paymentUnclear` 之类不在 27 码表 → 候选被丢 | 模型用驼峰自造码名 | `_coerce_code()` 加 `_snake()` + 21 条 `CODE_ALIASES` |
| 12 | `clausesId` 而非 `clause_id` | 模型键名走形 | `_canonical_keys()` 键名别名表递归改写 |

**第 7 条是最贵的一课**：`schema_hint` 看起来"加一句格式要求更稳"，实际是**把模型引向复述 schema**。
小模型对"形如 {...}"这类句式极其敏感——它会把括号里的键名当成待填清单，输出一片空骨架。

## 四、现在的实测结果

服务：

```
python -m zhilv_qiling_v2.local_llm --serve --port 8622 \
  --model-dir zhilv_qiling_v2/models/pretrained/legalone-1.7b --max-new-tokens 512
```

端到端（`samples/tech-development-contract.txt`，`ZLQL_LLM_MAX_CLAUSES=3`）：

```
logic_llm  | ok | 13451 ms | 模型遍历 3 条条款，采纳 1 条 / 引文未通过 1 条
judge_llm  | ok |  2640 ms | 复核意见已生成（分数仍由规则算：83）
```

| 项目 | 数值 |
|---|---|
| 模型加载 | 14.0 s（cuda） |
| 单条款生成 | 5.7 s（512 token，--no-think 时更快） |
| 3 条款端到端 | 13.4 s |
| 采纳率（引文校验） | 1/3 通过、1/3 被引文闸拦下、1/3 码不在表内 |
| 模型给出的码 | `amount_inconsistent`（通过）、`amount_missing`（码表外） |

**三条护栏仍然有效**：引文必须在条款里逐字定位（`_locate_quote`）、风险码必须在 27 码表内
（`_coerce_code`）、分数只读不写（`judge_llm` 只产复核意见）。

## 五、诚实的局限

1. **1.7B 会自造码名**。`amount_missing` 不在码表里就被丢弃，这不是模型"没发现问题"，
   而是它用了码表外的说法。可加别名，但不能让码表无限膨胀——码表是下游规则的契约。
2. **1.7B 的 message 常写成"条款缺失关键金额信息。根据…"这种自述**，`quote` 与 `message`
   容易黏在一起，引文校验会拦下。这是**设计上就该拦**，不是 bug。
3. **8B 没跑**。要 4bit 量化或 CPU offload，且本机显存空闲只有 5 GB。真要用 8B，
   建议走云上 vLLM 后按 OpenAI 兼容协议接（`ZLQL_LLM_ENDPOINT` 改一行）。
4. **`--no-think` 更快但更差**。关掉推理块后模型更容易直接吐空数组。
   日常用默认（带 think），只有批量跑时才考虑关。

## 六、复现命令

```powershell
# 1) 起服务
python -m zhilv_qiling_v2.local_llm --serve --port 8622 `
  --model-dir zhilv_qiling_v2\models\pretrained\legalone-1.7b --max-new-tokens 512

# 2) 指到服务上跑审查
$env:ZLQL_LLM_ENDPOINT='http://127.0.0.1:8622/v1'
$env:ZLQL_LLM_MODEL='legalone-1.7b'
$env:ZLQL_LLM_MAX_CLAUSES='3'
$env:ZLQL_LLM_TIMEOUT='300'
python -m zhilv_qiling_v2.cli review samples\tech-development-contract.txt --trace

# 3) 只看模型通道的原始回复（调提示词时用）
python tools\_llm_raw_dump.py
```
