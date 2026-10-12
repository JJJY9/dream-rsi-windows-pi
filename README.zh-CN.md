# Dream-RSI 第三节：Windows/Pi 独立复现

**语言：** [English](README.md) | 简体中文

本仓库在原生 Windows 上独立实现 [Dream-RSI v2 论文第三节](https://arxiv.org/abs/2609.14858v2)的核心循环，不是论文作者的官方代码。旧版本曾用真实 Pi 发现代理、模型编写的策略修订和固定评价器执行验收；新增的 v2 上下文与一致性实现另有[验收记录](docs/V2_ACCEPTANCE.md)，旧运行不能替代新代码的验收。DSH 适配器尚未完成真实模型验收。用户场景的实验独立设计，不要求复现论文第四节的基准实验。

当前公开内容是**核心算法和一个小型背包演示任务**。另行设计的 MOF 筛选实验准备采用更大的 worker 设置重新设计；此前的运行只作探索记录，本次发布不声称获得了 MOF 搜索收益。

## 算法流程

1. 发现代理修改候选程序；独立评价器评分，并将结果加入发现树。
2. 当前策略与模型编写的修订策略在已封存的历史世界上回放。回放复用已有节点和分数，**不会**再次调用发现代理或评价器。
3. 在论文规定的五个 `ρ` 设置上，依据 Eq. (1) 评价各策略；选取历史世界平均值最高的策略，部署到下一轮在线发现。

Eq. (1) 的值衡量历史世界中的发现质量与搜索代价，属于**内部策略选择指标**。它既不是候选程序的任务分数，也不保证策略在新世界中更好。符号、公式和代码对应关系见英文文档 [Algorithm](docs/ALGORITHM.md)。

## 快速运行

开发环境为 Windows 11 x64、原生 PowerShell、Python 3.11 或更新版本。离线测试不需要模型账号。

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m dream_rsi.cli --project-root . offline-test --outer-rounds 3
.\.venv\Scripts\python.exe -m pytest -q
```

`offline-test` 使用确定性测试代理运行完整循环，成功时报告 `OFFLINE_PASS`；这不是真实模型运行结果。

真实运行需要安装并登录 [Pi](https://github.com/earendil-works/pi)，并配置自己可用的 provider 和 model。以下命令会请求真实模型，可能产生费用：

```powershell
.\.venv\Scripts\python.exe -m dream_rsi.cli --project-root . model-check pi `
    --provider "YOUR_PROVIDER" --model "YOUR_MODEL"

.\.venv\Scripts\python.exe -m dream_rsi.cli --project-root . run `
    --backend pi --developer llm `
    --provider "YOUR_PROVIDER" --model "YOUR_MODEL" `
    --outer-rounds 3
```

默认运行仓库附带的背包任务。`--developer llm` 使用模型修订策略；`--developer scripted` 只是诊断用的固定程序，不能当作完整的真实模型运行。运行产物写在 `.runtime/`，不会进入 Git。

## 验证边界

英文 [Validation](docs/VALIDATION.md) 说明 Pi 路径经过了哪些独立核查、哪些尚未验证。实现保留第三节的 root/leaf 动作集合、只读可见前缀的历史回放、逐版本策略修订和 Eq. (1)。现有真实模型演示使用的是一个很小且容易饱和的任务，不能证明广泛任务上的自我改进收益。

模型生成的程序以当前 Windows 用户权限执行。进程限制和工作目录约定不能隔离文件系统或网络。运行不受信任的模型前，请阅读英文 [Security and execution](docs/SECURITY.md)。数据与软件来源见英文 [References](docs/REFERENCES.md)。

## 仓库内容

| 路径 | 用途 |
|---|---|
| `src/dream_rsi/` | 第三节控制器、发现树、回放、Eq. (1)、适配器和命令行接口 |
| `tasks/knapsack/` | 带固定评价器的小型演示任务 |
| `policies/` 与 `reference/` | 初始策略和独立数学参考实现 |
| `tests/` | 离线与回归测试 |
| `docs/` | 英文算法、验证、安全与来源文档 |

公开版本使用新的干净 Git 历史；内部开发指令、已被取代的验收报告、原始模型记录、本机运行数据和未完成的 MOF 实验不会包含在本次发布中。
