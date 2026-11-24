# GitHub 源码同步说明（2026-11-24）

本次从当前本地项目整理源码，在 `rain-pavilion/contract-review` 的 `main` 既有历史上形成新提交。原开发目录及其中未提交的工作、模型和业务数据保持原状；Git 远端改为当前有写权限的项目仓库。

## 发布范围

- 当前 `zhilv_qiling_v2` 应用、网页、测试和命令行，以及兼容的 `contract_review` 基线。
- 六节点协作与模型路由、来源绑定与逐项反馈、审查点框架、上下文组合检查、合同/法条/用户库功能。
- 启动脚本、部署配置、依赖清单、开发与使用文档、运行所需的小型法规/资质/案例元数据。
- 复现所需的工具源码与小型 JSON 参数；不包含深度模型权重。

## 未发布的本机资产

不包含 `.env`、密钥、私钥、用户账号与会话、合同原件、业务数据库、模型权重、本地推理运行时、虚拟环境、缓存、生成索引、原始调用日志、审查报告、标注数据、验收输出以及临时申报材料。名称包含“公开”的本地数据集也不自动视为本次发布对象。

历史 Markdown 文档可能提及 `tools/acceptance_results/`、`outputs/` 等本机留档路径；论文或申报工具也可能依赖未公开的模板和材料。现有工具源码不等于这些资产和运行证据已经公开，亦不能将工程测试视为法律质量成绩。本次保留仓库既有的历史工具文件，不更新其中的本机材料。

## 新版 CPU/离线起步

建议 Python 3.12；运行包含 JavaScript 检查的测试还需要 Node.js 22。

```text
python -m venv .venv
# 激活本平台的虚拟环境后：
python -m pip install -r deploy/requirements-deploy.txt
python -m zhilv_qiling_v2.cli doctor
python -m zhilv_qiling_v2.cli review samples/tech-development-contract.txt
python start_web.py --help
```

Windows 可使用 `start-offline-web.cmd`。完整法律模型审查需要另行准备模型权重、服务和对应配置；缺少这些资产时不能声称已经运行模型审查。

根 `requirements.txt` 保留旧版应用依赖。`deploy/requirements-web-model-locked.txt` 记录本机 CUDA 环境，不是跨平台安装保证，不与根依赖或 CPU 起步清单直接合装。

## 测试与证据边界

公开 CI 使用 CPU 依赖和 Node，执行应用自检、源码测试及法条库检查。只依赖本机未发布历史报告、训练输入或模型权重的回放用例，在所需资产缺失时显式跳过并记录原因；不得上传业务资产来消除此类跳过。

```text
python -m unittest discover -s zhilv_qiling_v2/tests_v2 -t . -v
python zhilv_qiling_v2/reports/experiments/demo_kb.py
```

工程测试通过只验证对应软件行为，不代表法律正确率、完整审查能力、模型训练收益或真实业务效率已经通过验收。

## 本次同步验证（2026-11-24）

- 本机 Python 3.12.14，在关闭 LLM 调用并启用离线模式的最终发布工作树中执行完整测试：`Ran 1369 tests in 83.251s`，`OK (skipped=34)`，退出码 0。
- 34 项跳过均有明确资产缺失原因，包括固定官方快照、历史训练对、真实模型输出、合同/扫描页及历史验收报告。未以合成结果替代这些证据。
- 环境自检、法条库检查、启动器帮助与公开样例的离线 CLI 审查均退出 0；法条库检查中未发布的真实合同语料部分跳过。
- 本轮未构建 Docker 镜像，也未在全新 Linux 环境独立安装验证；本地结果不代替 GitHub Actions 的实际执行结果。
