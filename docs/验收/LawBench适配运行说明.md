# LawBench官方协议固定与可续跑适配

本次两组实际固定前10题均已完成原版评分：默认模式与显式 `reasoning_effort=none` 均为 `score=0.0 / abstention_rate=0.0`，两组均10/10出现 `finish_reason=length`（128 token）。第二组结果保存在 `tools/acceptance_results/lawbench_1_2_first10_no_thinking.result.json`；第一组及其原始源码快照完整保留。请求级none模式未解决本组实验的输出截断问题，不能以开关配置代替质量验证，也不能把任一10题子集称为完整基准成绩。

已核验的官方源为 `https://github.com/open-compass/LawBench`，固定commit为 `e30981bb3ff54c41571f222e0b23e92d27375388`。官方README规定系统预测以 `系统名/任务ID.json` 保存，进入evaluation目录后执行 `python main.py -i 预测根目录 -o 结果.csv`。输出列为task、model_name、score、abstention_rate。任务1-1使用ROUGE-L，1-2使用Accuracy；不得用自造指标代替官方评分。

本地 `tools/acceptance_vendor/LawBench` 是不可变commit的稀疏官方源码快照，包含原版data与evaluation，未包含官方既有模型预测。Git smart HTTP在本次环境发生连接重置后，下载工具改用官方Git Tree API和固定commit的raw URLs。79个文件共36648470字节，每个文件分别核对官方Git blob SHA-1和SHA-256，`official_snapshot.json`保留来源、版本与hash。它不是自行生成的git历史，也不是模型评价结果。

快照与独立评价依赖均已通过局部.gitignore排除。可复现下载代码和协议manifest保留在源码中。运行时需要本地网络权限；已有快照不重复下载。

```powershell
python tools/acceptance_fetch_lawbench.py --dest tools/acceptance_vendor/LawBench
python tools/acceptance_lawbench.py plan --snapshot tools/acceptance_vendor/LawBench --out docs/验收/LawBench.manifest.json
```

原版数据含zero_shot和one_shot各20任务，每任务500条。plan实际加载全部40个文件并检查instruction/question/answer字段、数量和hash。one_shot的演示内容已在官方instruction中，不自行编造或抽取测试答案做示例。

## 真实模型推理与中断续跑

`infer`向显式配置的本地原生服务发送官方instruction加换行及question；当前题的answer不会发给模型，不添加自写答题提示。当前验收入口先读取`/v1/models`，要求精确别名`legalone-8b`、llama.cpp后端及实际8195978240参数，并逐题核对响应中的model。`--model-revision`是操作者声明的版本或文件hash，实际原生架构元数据另存；远程接口不能据此证明所加载权重的文件hash。必须结合本机权重来源与实际文件摘要留证。

每次请求前先将独立调用收据持久化为started，结束后记录成功或失败、响应模型、usage、finish_reason、耗时、配置hash和身份元数据摘要，文件位于`receipts/任务ID/题序-call_id.json`。成功收据先完整提交，再写官方prediction格式和state；崩溃后可恢复已经落盘的真实响应，不能用假答案补齐。每个新run自动保存按字节复制的`adapter_snapshot.py`及其hash，源码或快照漂移时拒绝续跑。

相同配置重跑只补固定范围内尚未成功的题；修改模型、shot、任务范围、推理模式、温度、输出长度或适配器源码必须使用新run目录。已明确失败的请求保留收据后可以重试；只有started而没有结束结果的调用视为结果未知，工具拒绝自动重复推理，需要先核实中断情况。不能删除收据或改写旧预测来绕过续跑检查。

下面命令只生成新的1个响应用于连通性与协议核验。MODEL_REVISION_OR_WEIGHT_HASH必须替换成实际模型固定版本或权重hash；不能把任意名称当作版本证明。工具不会因为health配置为enabled就宣称实际生成。

```powershell
python tools/acceptance_lawbench.py infer --snapshot tools/acceptance_vendor/LawBench --run-dir tools/acceptance_results/lawbench_run_zero --endpoint http://127.0.0.1:8620/v1 --model legalone-8b --model-revision MODEL_REVISION_OR_WEIGHT_HASH --shot zero_shot --tasks 1-2 --max-items 1 --max-tokens 512 --timeout 600
```

完整单shot必须覆盖20任务各500条。明确传`--tasks all --max-items 0`才持续处理所有剩余条目；默认每次只生成1条，避免在CPU环境误启动长时间全量推理。one_shot使用另一个run目录，保留两种协议边界。输出长度截断、提示模板、推理参数和模型版本应在比较时一并报告。

## 原版评价器执行

```powershell
python tools/acceptance_lawbench.py evaluate --snapshot tools/acceptance_vendor/LawBench --run-dir tools/acceptance_results/lawbench_run_zero --python .venv/Scripts/python.exe
```

工具在调用前后都核对原版快照，检查输出item IDs、原始prompt和reference与官方数据严格对应，并核对保存的适配器源码、预测文件hash和逐题成功收据，然后从evaluation工作目录调用未修改main.py。一次评价只允许state声明的单一模型目录，禁止把多个模型的题数累计成完整成绩。少量任务或少量题的结果始终标为subset；只有同一模型在20个官方任务上各有500条实际输出，才可标完整单shot。默认拒绝覆盖旧CSV，避免把旧分数误归因到新的预测。

官方依赖包含旧版本软件，其中python-Levenshtein 0.21.1在本机Python3.12构建失败。最小1-2评分实际仅需rouge_chinese、jieba、nltk、cn2an及pandas等导入依赖，已安排安装到独立eval_deps目录，不修改模型环境。完整任务2-1的原版ChERRANT调用python3且依赖LTP资源，完整正式评价需要另外配置兼容执行环境，不能静默跳过或替换其评分函数。

## 适配器集成检查与成绩边界

`fixture`只为官方1-2前3题写入故意为空的prediction，声明fixture_only、model_calls=0，用于验证官方main.py的弃权处理和数据对齐，不是本项目模型输出，也不是模型成绩。它不把reference复制成预测，不通过伪造高分验证成功。

```powershell
python tools/acceptance_lawbench.py fixture --snapshot tools/acceptance_vendor/LawBench --run-dir tools/acceptance_results/lawbench_adapter_fixture
python tools/acceptance_lawbench.py evaluate --snapshot tools/acceptance_vendor/LawBench --run-dir tools/acceptance_results/lawbench_adapter_fixture --python .venv/Scripts/python.exe --out tools/acceptance_results/lawbench_adapter_fixture.result.json
```

正式全量模型评价仍为not_run；当前已完成官方协议核验、固定数据/评价源码准备、可续跑入口，以及下述官方1-2固定前10题的真实模型子集评分。子集运行不能替代完整任务500题或全量20任务。ProvBench已另外核验ACL2025论文及labpaper/ProvBench代码脚注，但该公开仓库本次HTTP404，数据/原版评价器不可获取；不能将LawBench或自建合同冒充其官方数据，详情见ProvBench适配运行说明。

本次空输出fixture已由原版main.py实际执行成功：exit_code=0，3题官方1-2，score=0.0、abstention_rate=1.0，符合预期；前后官方源码hash一致。证据为tools/acceptance_results/lawbench_adapter_fixture.result.json及其official_results.csv。初次普通沙箱进程不能读取提升权限安装的eval_deps，评分需在可读取该目录的执行上下文运行；不通过改写官方评分器绕过依赖。

## 已完成的默认模式官方1-2固定首10题

本次原始组使用LegalOne-8B Q4_K_M、本地llama.cpp、zero_shot任务1-2固定前10题、temperature=0、max_tokens=128，没有在请求中传入reasoning_effort。原版评分器实际返回`score=0.0`、`abstention_rate=0.0`，`full_official_benchmark=false`。这是10题上的原始输出成绩，不是500题任务成绩、全量成绩或合同审查准确率。

逐题收据表明10/10响应均为`finish_reason=length`，completion_tokens均为128。输出在分析、题意或格式复述中被截断，尚未形成可靠的最终答题输出；其中个别响应复述了`[正确答案]`格式甚至示例，不能简单说所有输出都没有该标记。原始响应未经清洗、重写或选择性提取，直接交给原版评分器。零分必须连同这一输出配置和截断事实报告，不能用适配器连通成功掩盖，也不能据此估计模型完整能力。

原始证据保留在`tools/acceptance_results/lawbench_1_2_first10.result.json`、`lawbench_run_zero_1_2_first10/official_results.csv`、该目录的prediction、10份逐题收据和state。该次实际适配器原样保存为`lawbench_run_zero_1_2_first10/adapter_snapshot.py`，SHA-256为`3239c0cbdb2528dc412ad3cdade10803f6f92ef8ce9b224b732840dc56a53323`，与原state记录一致。

下面是已完成组的原始配置记录，不应以新版适配器重新写入该目录。需要重现时使用另一个新目录并保留其自身源码快照。在PowerShell中先将`$actualGgufFile`指向实际由服务加载的GGUF文件，再计算真实文件摘要；本项目端点为8620/v1。不要在服务未ready时启动试验。

```powershell
$lawbenchModelRevision = 'sha256:' + (Get-FileHash -LiteralPath $actualGgufFile -Algorithm SHA256).Hash.ToLowerInvariant()
& .venv/Scripts/python.exe tools/acceptance_lawbench.py infer --snapshot tools/acceptance_vendor/LawBench --run-dir tools/acceptance_results/lawbench_run_zero_1_2_first10 --endpoint http://127.0.0.1:8620/v1 --model legalone-8b --model-revision $lawbenchModelRevision --shot zero_shot --tasks 1-2 --item-limit 10 --max-items 10 --max-tokens 128 --temperature 0 --timeout 600
& .venv/Scripts/python.exe tools/acceptance_lawbench.py evaluate --snapshot tools/acceptance_vendor/LawBench --run-dir tools/acceptance_results/lawbench_run_zero_1_2_first10 --python .venv/Scripts/python.exe --out tools/acceptance_results/lawbench_1_2_first10.result.json
```

`--item-limit 10`限定子集范围，`--max-items 10`限定本次最多新增响应；在适配器和配置均未改变、收据完整的前提下，中断后同命令只补足剩余题，不会扩展到第11题。已经完成的旧组和已生成CSV均保留，不能用新组覆盖。

## 显式关闭thinking的同一首10题对照运行

本机固定版本llama.cpp的`tools/server/README.md`记载请求参数`reasoning_effort=none`可关闭reasoning/thinking。适配器提供可选`--reasoning-effort`：省略时不传入该字段，也不向原配置添加空值；显式指定时仅作为HTTP运行参数，纳入配置hash和逐题收据。官方instruction、question、答案和评分协议保持不变。

对照组使用相同官方zero_shot任务1-2前10题、同一模型与128 token上限，另建`lawbench_run_zero_1_2_first10_no_thinking`目录。该组必须完整保留全部10题及其失败、截断或低分结果，与原始默认模式组并列展示，不能挑选较好的一组代替原组。下列是独立运行入口，本说明不预填该组分数；须等待真实推理和原版评分器结束后再登记。

```powershell
$lawbenchModelRevision = 'sha256:' + (Get-FileHash -LiteralPath $actualGgufFile -Algorithm SHA256).Hash.ToLowerInvariant()
& .venv/Scripts/python.exe tools/acceptance_lawbench.py infer --snapshot tools/acceptance_vendor/LawBench --run-dir tools/acceptance_results/lawbench_run_zero_1_2_first10_no_thinking --endpoint http://127.0.0.1:8620/v1 --model legalone-8b --model-revision $lawbenchModelRevision --shot zero_shot --tasks 1-2 --item-limit 10 --max-items 10 --max-tokens 128 --temperature 0 --reasoning-effort none --timeout 600
& .venv/Scripts/python.exe tools/acceptance_lawbench.py evaluate --snapshot tools/acceptance_vendor/LawBench --run-dir tools/acceptance_results/lawbench_run_zero_1_2_first10_no_thinking --python .venv/Scripts/python.exe --out tools/acceptance_results/lawbench_1_2_first10_no_thinking.result.json
```
