# Mamba 实际训练与显式接入

本产物是轻量实际训练和工程连通验证。训练数据是合成规范及弱监督标签，不是现行法条、独立人工金标或正式基准评测，不能据此声称充分训练或申报指标达标。默认 `models/bert_entail` 和原权重保持不变。

## 已完成的训练

- 执行命令：在项目根目录运行 `.\.venv\Scripts\python.exe tools\train_mamba_entail.py`。
- 288 条合成文本对、24 个来源/模板组，按整组拆分为训练 216、验证 36、测试 36；每个拆分包含 entail/contradict/neutral 三类。
- 冻结旧 BERT、交叉注意力和归一化层，实际训练 Mamba 选择性状态空间核心、768→64→768 投影、门控和新的分类头；本次禁用规则残差。
- 真实执行 3 epochs、162 optimizer steps。epoch 平均 CE 为 0.909209、0.342493、0.163273，训练 9.839 秒，整个训练/保存/重载/预测流程 31.390 秒。
- Mamba 的 A_log、D、x_proj、dt_proj 等状态空间参数真实变化，162 步均有非零梯度。GPU 峰值 allocated 503,357,440 bytes、reserved 549,453,824 bytes。训练子进程已退出并释放其 CUDA context。
- 新模型已 strict 重新加载并执行三次真实预测。弱合成测试集 36 条的 macro-F1=0.7693 仅作诊断，不能作为人工标注成绩。
- 完整可复核证据：`tools/acceptance_results/mamba_entail_training/run.json`、`pairs.jsonl`。独立新权重：`zhilv_qiling_v2/models/bert_mamba_entail/model.pt`。

## 在新网页进程中显式选择

本机已有新权重时，可直接使用项目根目录的 `start-core-web.cmd`，同时启动或复用LegalOne-8B并显式选择本分支；原基础入口继续保留旧模型选择规则。

以下 PowerShell 命令从项目根目录执行。应先结束需要更换模型的旧网页进程，再启动新进程；健康检查不会热切换或重新加载模型。端口可以按本机实际情况调整。

```powershell
$env:ZLQL_ENTAILMENT_MODEL_DIR = (Resolve-Path '.\zhilv_qiling_v2\models\bert_mamba_entail').Path
$env:ZLQL_ENTAILMENT_BACKEND = 'bert'
$env:ZLQL_ENTAILMENT_DEVICE = 'cpu'
.\.venv\Scripts\python.exe start_web.py --port 8611 --no-browser
```

这三项仅选择深度法条判别分支；LegalOne-8B 的真实服务连接仍按主项目的启动和配置方式单独进行。CPU 选择可以避免与 8B 争用有限显存。

`GET /api/health` 的 `runtime.bert_metadata` 提供以下实际加载信息：

- `loaded_checkpoint.directory` 与 `weights_file`：实际选择的绝对路径；
- `loaded_checkpoint.selection`：环境变量显式选择为 `environment`，缺省为 `legacy_default`；
- `loaded_checkpoint.supervision`、`weak_supervision`、`human_gold`、`training_scope`：训练来源与适用限制；
- `loaded_architecture.mamba`、`mamba_config`、`strict_checkpoint_load`：实际构造和严格加载的架构。

显式路径不存在或权重不兼容会报错，调用方会可见地降级为 rules，绝不自动替换成旧 BERT。`load_predictor()` 仅按解析后的路径缓存加载成功的对象；失败不会缓存为成功。上层服务的降级状态在一次进程生命周期内保持稳定，修复权重后应重新启动，或由维护入口显式重新初始化。

恢复默认模型：

```powershell
Remove-Item Env:ZLQL_ENTAILMENT_MODEL_DIR -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe start_web.py --port 8611 --no-browser
```

## 可复现检查

```powershell
.\.venv\Scripts\python.exe -m unittest zhilv_qiling_v2.tests_v2.test_mamba_training -v
.\.venv\Scripts\python.exe tools\verify_mamba_web_runtime.py
```

第二条会启动隔离的 CPU 网页进程、使用临时数据库及随机回环端口，读取真实 HTTP 健康结果，并在同一进程调用三次新分支预测。它关闭 LLM、NER、BGE，不操作 8610/8620，不把这些合成预测当端到端合同验收或正式模型效果。证据写入 `tools/acceptance_results/mamba_entail_training/web_runtime.json`，随后退出并释放资源。
