# 小项目适配调研：Dream-RSI 机制能接到哪些 GitHub 小项目上

- 日期：2026-10-13
- 目的：MiQi 暂缓；转而在 GitHub 上寻找**小项目**，评估 Dream-RSI §3 机制（在线探索树 + 历史回放模拟器 + 策略自改进）的适配度。
- 判据（来自本仓库四轮真实实验的经验）：
  1. **客观评估器**：可运行、确定性、输出标量分数（与 agent 隔离）；
  2. **可爬坡梯度**：目标不饱和或至少有成本维度的权衡空间（几十到几百次尝试的余地）；
  3. **树语义成立**：尝试可组织为"根 + 分支 + 精修链"（同一工作区的多轮改进）；
  4. **本地可跑**：纯 CPU、依赖轻、单次评测秒级到分钟级。

## 一、Tier 1：即插即用级（评估器现成）

### 1. OpenEvolve 示例任务集 ★7.5k（首推）
`algorithmicsuperintelligence/openevolve` 的 `examples/` 下有 **24 个自包含任务**，每个包含：初始程序 + 固定 `evaluator.py`（加载候选程序、超时保护、输出 score）+ 目标值。纯 Python 可跑的含：
`circle_packing`（**正是论文 §4 同款任务族**，n=26，目标 2.635）、`function_minimization`、`tsp_tour_minimization`、`signal_processing`、`competitive_resource_allocation`、`symbolic_regression`、`online_judge_programming`、`llm_prompt_optimization` 等。

- **接入成本**：每个任务写 ~30 行包装（把 evaluator 的 score 转成我们 `tasks/` 协议的 JSON）即可进 `FixedCommandEvaluator`。
- **额外价值**：openevolve 自身就是"发现循环"框架（且已带 `claude_code_quickstart`——生态同样在用成熟 agent 当编码器）。可以在同一 evaluator 上做**机制对照**：openevolve 的进化循环 vs Dream-RSI 的树+回放选择循环。
- **适配度：★★★★★**

### 2. Neuralink 无损压缩挑战（`phoboslab/neuralink_brainwire` ★90、`mikaelhaji/n1-codec` ★47）
固定 146MB 神经信号语料；`eval.sh` 输出**压缩比**（无损校验 + 比率 = 完美评估器）；当前公开最好 ~3.35x，作者论证 ~4x 可能是天花板。
- **回放语义完美**：多个压缩器尝试共享同一语料，历史树=不同编码策略的实测结果。
- **特殊价值**：这是"薄梯度/近饱和"任务的天然样本——可以观察机制在接近天花板时的行为（选型是否及时收敛/停滞，对应我们实验中观察到的饱和惰性）。
- **适配度：★★★★☆**（评估器满分；梯度可能有限）

### 3. modded-nanogpt speedrun（`KellerJordan/modded-nanogpt` ★3k+）
"90 秒把 124M GPT 训到目标 loss"——社区多年持续改进，是**不饱和金标准**；评估器=定时训练脚本（确定性误差范围内）。
- **门槛**：需要 GPU（本机无）。**如果组里有 GPU 机器，这是第一推荐。**
- **适配度：★★★★★（有 GPU 时）**

## 二、Tier 2：适配但要小改造

### 4. 游戏类 bot（2048 / Snake / Sudoku 求解器）
评估器=对局得分/求解时间，天然标量且长梯度（启发式+搜索深度可无限打磨）。
- 注意：**必须固定种子**才能做确定性回放；`kcwu/2048-python` ★78 虽是现成的竞赛框架（甚至内置 `average_score/code_size` 双目标——与 Eq.(1) 的质量-成本权衡同构！），但其 harness 是 2014 年的 telemetry 浏览器驱动，已不可用——需自写 ~100 行对局模拟器。
- **适配度：★★★★☆**（自写评估器约半天）

### 5. 性能优化小库（pytest + benchmark 的纯 Python 库，如 JSON parser、bitset、LRU cache）
评估器=测试通过 + 计时（需多次重复、取中位数抗噪）。
- 梯度不饱和（perf 可长期打磨）；但**评测噪声**是确定性回放语义的敌人，需要较强的统计包装。
- **适配度：★★★☆☆**

## 三、Tier 3：生态对照（非"小项目"）

`openevolve` / `ThetaEvolve` / `EvoControl` / `funsearch` 复现 —— 同为发现循环框架。可做"**同一任务、两种循环机制**"的受控对照（研究向），或作为任务来源（其 examples 即 Tier 1）。

## 四、结论与建议的下一步

- **最优先**：把 openevolve 的 `circle_packing` + `tsp_tour_minimization` 两个任务接入本 fork（半天工作量），在新的第三方评估器上验证机制泛化——这比再自造任务更有说服力；
- **次优先**：压缩挑战作为"近饱和"对照组跑一轮；
- 有 GPU 环境时：speedrun 是机制价值最大化的场景（长时程、大预算、探索策略决定性最强）。

*评估依据：本仓库在 packing 任务上的四轮真实实验 + 本调研中对各仓库 evaluator/README 的逐一核验（2026-10-13）。*
