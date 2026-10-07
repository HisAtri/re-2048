# re-2048

2048 的自对弈强化学习工程。C++ 负责规则、搜索、自对弈与数据落盘，PyTorch 负责神经网络
的训练与推理，两者之间是**批**推理协议而不是逐样本调用。

## 快速开始

```bash
uv tool install cmake && uv tool install ninja && sudo apt-get install -y ccache
make env       # 创建虚拟环境并安装依赖（含 torch）
make build     # 构建 C++ 内核、CLI、测试与 Python 扩展
make test      # CTest 六个套件 + pytest

make run ARGS="version"
make run ARGS="generate --policy random --samples 200000 --seed 20261006 --out /tmp/random_v1"
make run ARGS="verify --data /tmp/random_v1"
```

完整说明见 [docs/BUILD.md](docs/BUILD.md)。

## 工程结构

```
cpp/
  core/     参数解析、日志、文件、JSON 子集、哈希、路径
  game/     棋盘表示与查表移动生成、规则、种子派生
  search/   批量 NN-MCTS 引擎、UCT+rollout 搜索
  dataio/   训练样本记录、分片写入、清单与来源校验
  program/  对局驱动、纯 C++ 跑谱、评测、校验
  command/  CLI 子命令
  ext/      pybind11 绑定（导入名 r2048_core）
python/src/r2048/
  nn/       双头网络与棋盘编码
  runtime/  批推理服务与自对弈驱动
  reference/ 规则语义的参考实现（仅测试使用）
docs/adr/   架构决策记录
```

## 关键设计（都记在 ADR 里）

* **动作顺序是硬契约**：`UP=0, RIGHT=1, DOWN=2, LEFT=3`。策略头四条输出、数据记录里的
  访问数顺序都按它排列，见 [ADR 0004](docs/adr/0004-rules-and-action-order.md)。
* **棋盘用 16 个 u8（不设指数上限）+ 半行查表**做移动生成，跨半区合并的优先级是实现的
  风险点，由穷举与随机测试守住，见 [ADR 0005](docs/adr/0005-board-representation-and-tables.md)。
* **双头网络**：价值头看 afterstate `u = T(s,a)`，策略头看决策局面 `s`；搜索展开一个决策
  节点时同时需要 1 个策略结果与最多 4 个价值结果，因此一次前向天然覆盖 5 倍以上的局面，
  见 [ADR 0003](docs/adr/0003-cpp-python-boundary.md)。
* **数据可重放**：数据文件逐字节只由 `(种子, 参数, 目标条数)` 决定；
  `r2048 verify` 会用清单里的参数重新生成整批数据并逐分片比对哈希，
  见 [ADR 0007](docs/adr/0007-data-format-and-provenance.md)。
* **规则只实现一次**：Python 侧只有一份仅用于测试的参考实现，跨语言一致性由
  向量文件比对保证（`make test` 的一部分）。

## 现在的状态

**已完成**：构建体系、规则引擎与查表移动生成、纯 C++ 跑谱（随机 / UCT-rollout）、
数据格式与校验、批量 NN-MCTS 引擎、Python 批推理与自对弈驱动、双头网络与编码、
数据层（按局划分）、训练器（value / policy(冻结价值) / joint / dual）、
评测（三条 track + 配对 bootstrap）与推理探针。

**第一轮实测**（数字与命令见 [docs/BUILD.md](docs/BUILD.md)、口径见
[ADR 0010](docs/adr/0010-training-recipes-and-gates.md)）：

| 对照 | 结果 |
|---|---|
| nn_value − 随机 | **+1916.9**（95% CI [+1706, +2134]）→ 通过 |
| nn_policy − 随机 | +212.7（95% CI [+75.6, +356.8]）→ 显著但很弱 |
| NN-MCTS64 − 纯 MCTS64 | −1812.8（95% CI [−3688, +83]）→ 未超过纯搜索 |
| 候选监督后：NN-MCTS64 − 纯 MCTS64 | **+38.9 评分点**（同上评测口径） |

最后一行与旧版同一阶段一致：差距来自**弱教师与数据量**，而不是管线。

**候选级监督已实现并验证**（[ADR 0015](docs/adr/0015-candidate-level-supervision.md)）：
价值头训练在搜索实际查询过的**全部合法候选**上（目标 `A(u_a) = Q̄(a) − r(s,a)`），
按 `sqrt(访问数)` 加权。它同时修掉了两个已被实测记录的问题：
“价值头在自己被查询的地方没有监督”（同一教师数据下评分 858 → 1398）与
“自举迭代把价值标度训坏”（轮间从 **−279.5** 变成 **+125.2**）。
同一套两轮迭代现在表现为**变强**：NN-MCTS-64 评分 967.7 → **1038.9**，
首次高于同预算的纯 MCTS-64 锚点（1000）。

**搜索预算与协议变量的实测**（[ADR 0016](docs/adr/0016-protocol-variables-and-simulation-budget.md)）：
模拟数 → 评分曲线在 64/128/256/512/1024 上完整测过，**两条线到 1024 都还在涨**——
纯 rollout 是 1000.0 / 1082.4 / 1137.7 / 1183.8 / 1204.6，NN 引导是
1050.2 / 1138.3 / 1180.6 / 1209.9 / **1236.9**（每翻倍约 +21~+88 分，秒/局涨 2.1~2.3×）。
**等预算下 NN 指引全线优于纯 rollout**（+26~+56 分，区间都不含 0），且大致等价于
**把搜索预算放大一倍**（NN-512 ≈ 纯-1024）。评测/自对弈口径的性价比拐点在 128~256。
但**候选标签的量级不随纯 rollout 教师预算变化**（263 / 288 / 282 分）——
抬高价值目标要换教师语义（NN 引导搜索当教师 = 420 分），不是加预算。
另外发现一个此前没被当成协议的量：NN 路径的**批组成**（`lanes/in-flight/max-batch/cache-entries`，
乃至 `device`）会改变逐局结果，只改 lane 数就有 +20 分的位移，因此必须冻结并进协议指纹。
这条的归因经过一次订正——**不是浮点，而是在途模拟把搜索切分成了不同的序列**；
而且这条为了吞吐而引入的在途流水线有实质强度代价（`in_flight` 1→4 时均分 12522→8923、
到 2048 的局 9.4%→0%），见 [ADR 0017](docs/adr/0017-batch-scheduling-dependence.md)。
而评测 seed 恰恰**不该**进协议指纹（它只是同一协议下的重复抽样）。
后两者是待修的口径问题，见 ADR 0016 决策 5/6。

**尚未实现**：训练断点续跑的完整语义（当前只保存/恢复优化器与轮次）、模型注册与发布、
机器画像、GUI。
