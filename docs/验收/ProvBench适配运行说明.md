# ProvBench官方身份与本地适配

ProvBench已由ACL Anthology官方论文确定身份：`2025.acl-long.312`，题名为《ProvBench: A Benchmark of Legal Provision Recommendation for Contract Auto-Reviewing》，DOI为`10.18653/v1/2025.acl-long.312`。实际全文第一页代码脚注指向`https://github.com/labpaper/ProvBench`，此前候选`LittleFishG/ProvBench`不是该论文脚注地址。身份核验不再是缺项；公开仓库能否访问是独立问题。

`tools/acceptance_provbench.py probe`实际下载ACL页面及完整PDF，读取全部页面及PDF链接注解，保存题名、作者、DOI、SHA-256、核验时间和HTTP状态到`ProvBench来源核验/source_check.json`。论文身份以实际下载的主源证据为准，候选标题仅用于检索。完整PDF与HTML缓存不进入源码版本控制。页码证据索引已从SHA-256保持一致的缓存PDF补录，索引更新时间与原始HTTP核验时间分别记录。

## 已核验的论文协议

论文包含两项任务：为合同条款推荐法律条文，以及判断条款与条文的entailment/contradiction关系。PROVDATA为3550条款，每条款3个黄金条文，候选库663条；原划分为训练2698、验证404、测试448条款。数据来源为§3与表1/表2，表2位于印刷页6243（PDF第4页）。训练和测试的条文对数量分别为8094、1344，不能将条文对数量误作合同数量。表2验证集的两类计数589+704=1293，与404条款乘3得到的1212不一致；当前保留论文原数并记录待官方数据核对，不自行改表或捏造修正版。

评价定义的原文出处为§5.2，印刷页6245（PDF第6页），已逐页读取全文并视觉核对该页：R@1/R@3/R@5是前k个候选中至少命中一个相关条文的条款比例，而不是命中条文数除以3。Top-3 EM要求推荐的前三个条文与黄金集合完全相同，本地诊断使用大小为3的集合相等比较，不因排列顺序变化而判错。冲突检测使用每条款的三个黄金条文进行关系判断，从而与检索错误分开；报告entailment和contradiction的类内accuracy及overall F1。论文未将MRR/NDCG列为正式指标，工具不会用它们替代论文口径。

§5.2仅写overall F1，没有给出正类选择、micro/macro/weighted averaging公式。原版公开代码不可取时不能补猜这个精确定义。工具只提供明确命名的macro/micro/contradiction-positive F1诊断，`official_overall_F1=null`，待原版评价实现可获得后再确定正式口径。附录C（印刷页6254、PDF第15页）明确数字0表示无语义矛盾、1表示潜在冲突；适配层使用命名标签避免与项目风险类别混淆。上述诊断不伪装成原版官方评价器分数。

## 目前具体缺项

官方论文已确认，不能继续写“名称不唯一”。本次官方仓库网页、main README和master README三处实际HTTP均为404。运行时缺的是真实可获取的官方PROVDATA原文件、663项候选库、原始split IDs与文件版本、发布格式映射、官方评价代码及具体入口，以及实际模型预测与模型版本。仓库HTTP状态以本次source_check记录为准；404只表示当时公开地址不可访问，不能断言仓库从未存在，也不能据此改变论文身份。

如有真实官方本地checkout，可在manifest中填写其origin、commit、未修改评价脚本hash、工作目录和参数；`run-official`会检查固定版本和数据/预测/模型条件后执行实际官方脚本。未补齐时严格返回not_run，不从项目自己的法条库凑663条，不自行生成“官方”测试集，不把合成候选当作PROVDATA。

```powershell
python tools/acceptance_provbench.py probe --manifest docs/验收/ProvBench.manifest.json --sources docs/验收/ProvBench来源核验
python tools/acceptance_provbench.py check --manifest docs/验收/ProvBench.manifest.json --out tools/acceptance_results/ProvBench.not_run.json
python tools/acceptance_provbench.py run-official --manifest PATH_TO_COMPLETED_REAL_MANIFEST.json --python .venv/Scripts/python.exe --out NEW_OFFICIAL_RESULT.json
```

## 本地交换格式与可执行诊断

交换格式是本项目适配层格式，不声称为尚未取得的官方发布JSON schema。必须在manifest登记原始官方文件SHA-256、转换脚本SHA-256及原始ID/split映射，才能对真实数据执行诊断。

- 候选库JSONL：每行`provision_id`及`text`，必须663项且ID唯一。
- 测试gold JSONL：每行`id`、`official_split=test`、`clause`、`gold_provisions`；最后一项为3个`{provision_id,label}`对象，label为`entailment`或`contradiction`。
- 预测JSONL：每行相同`id`，`ranked_provision_ids`为至少5个不重复候选ID；`conflicts_on_gold`是黄金3个条文ID至关系预测的映射，不能用检索到的3个条文替代。

`evaluate`核对448条测试条款、663候选、ID完整覆盖、黄金3条及关系标签后计算论文指标诊断；由于尚未运行原版代码，其结果始终`official_evaluator_executed=false / formal_score=false`。

```powershell
python tools/acceptance_provbench.py evaluate --manifest PATH_TO_COMPLETED_REAL_MANIFEST.json --out NEW_DIAGNOSTIC_RESULT.json
python tools/acceptance_provbench.py fixture --fixture-dir tools/acceptance_results/provbench_fixture --out tools/acceptance_results/provbench_fixture.result.json
python tools/acceptance_provbench_tests.py
```

fixture只含2条合成计分样例和5个合成候选，用于验证R@k命中口径、Top-3 EM、关系混淆矩阵及黄金条文分支；明确不是官方数据、没有模型调用、不是模型成绩。未写入任何人工标注身份。正式ProvBench成绩仍为not_run。
