# 构建与运行手册

## 目录与约定

| 位置 | 默认值 | 说明 |
|---|---|---|
| 构建目录 | `$HOME/r2048/build` | CMake 构建树与全部产物（`bin/`、`lib/`、`pyext/`） |
| 虚拟环境 | `$HOME/r2048/venv` | uv 管理；仓库内有一个指向它的 `.venv` 符号链接 |
| 输出根 | `$HOME/r2048/output` | 跑谱数据、模型、评测结果（`R2048_OUTPUT_DIR` 可覆盖） |

三者都在本地文件系统上，不放仓库里（仓库位于 9p 挂载，编译与大规模写盘都更慢）。
`make env-check` 会打印实际取值。

## 首次准备

```bash
uv tool install cmake     # 22.04 自带的 3.22 太旧；uv 装的是新版且不需要 sudo
uv tool install ninja
sudo apt-get install -y ccache    # 可选，只影响重复编译速度

make env      # uv sync：装 torch（cu130 wheel）等依赖，创建 .venv 符号链接
make build    # 配置 + 构建库、CLI、测试、Python 扩展
make test     # CTest 六个套件 + pytest
```

## 日常命令

```bash
make build                       # 增量构建（改一行 C++ 只需一次 ninja 链接）
make test                        # 全部测试
make test-cpp                    # 只跑 C++ 套件
make test-py                     # 只跑 Python 测试（含跨语言一致性）
make fmt                         # clang-format（缺省时跳过）+ ruff

make run ARGS="version"                                  # 运行 C++ CLI
make run ARGS="generate --policy random --samples 200000 --seed 20261006 --out /tmp/random_v1"
make run ARGS="evaluate --policy mcts --simulations 64 --games 200 --seed 20262052"
make run ARGS="verify --data /tmp/random_v1"             # 含重放校验

make py ARGS="version"                                   # Python 侧入口
make py ARGS="selfplay --checkpoint runs/x/model.pt --samples 100000 --simulations 64"
make py ARGS="train --data <数据批> --output <run 目录> --mode value --epochs 20"
make py ARGS="eval --checkpoint <模型> --track nn_mcts --games 200 --seed 20262052"
make py ARGS="compare --baseline a.jsonl --candidate b.jsonl"
make py ARGS="probe --checkpoint <模型>"                  # 打印单局面的网络输出（推理自检）
```

## 从零跑一轮的最小流程

```bash
# 1) 跑谱：随机策略数据（约 45 万样本/秒）
make run ARGS="generate --policy random --samples 1000000 --seed 20261007"
# 2) 训练价值头（约 22 万样本/秒）
make py ARGS="train --data $R2048_OUTPUT_DIR/data/<批> --output $R2048_OUTPUT_DIR/runs/value-1 --mode value --epochs 8"
# 3) 基线评测（纯 C++，同一 seed 才能按局配对）
make run ARGS="evaluate --policy random --games 200 --seed 20262052 --jsonl $R2048_OUTPUT_DIR/eval/random.jsonl"
make run ARGS="evaluate --policy mcts --simulations 64 --games 200 --seed 20262052 --jsonl $R2048_OUTPUT_DIR/eval/mcts64.jsonl"
# 4) 网络评测 + 配对比较
make py ARGS="eval --checkpoint $R2048_OUTPUT_DIR/runs/value-1/model.pt --track nn_value --games 200 --seed 20262052 --jsonl $R2048_OUTPUT_DIR/eval/nn_value.jsonl"
make py ARGS="compare --baseline $R2048_OUTPUT_DIR/eval/random.jsonl --candidate $R2048_OUTPUT_DIR/eval/nn_value.jsonl"
# 5) 策略头：先用 MCTS 数据当教师，再在冻结价值上训策略头
make run ARGS="generate --policy mcts --simulations 64 --samples 200000 --seed 20261008"
make py ARGS="train --data $R2048_OUTPUT_DIR/data/<mcts 批> --output $R2048_OUTPUT_DIR/runs/policy-1 --mode policy --freeze-value --init-checkpoint $R2048_OUTPUT_DIR/runs/value-1/model.pt --epochs 15"
```

路径可用环境变量覆盖：`make BUILD=/data/build build`、`R2048_OUTPUT_DIR=/data/runs ...`。

## C++ CLI 一览

| 子命令 | 作用 |
|---|---|
| `generate --policy random\|mcts` | 自对弈跑谱，产出带来源记录的数据批 |
| `evaluate --policy random\|mcts` | 固定局数评测，输出逐局 JSONL 与摘要 |
| `verify --data <批>` | 结构 + 摘要校验，并（默认）用清单参数重放比对哈希 |
| `version` | 规则/协议版本与构建身份（JSON） |

约定：所有参数都是具名的（`--key value` 或 `--key=value`），未知参数直接报错；
stdout 只放可被脚本消费的结果，进度与统计走 stderr。

## Python 侧

```
python -m r2048 version          # 两侧版本、构建身份、torch/CUDA 状态
python -m r2048 env              # 输出根等路径
python -m r2048 selfplay ...      # NN 引导的批量自对弈（需要 --checkpoint，或用 --random-init 自检）
```

扩展模块 `r2048_core` 由 CMake 构建到 `build/pyext/`，通过 `PYTHONPATH` 引入
（Makefile 已代劳）。它不在 PyPI 上，也不会被装进 site-packages——源码树里因此
不会出现任何二进制产物。

## 性能记录（本机实测，16 核 WSL2 + 直通 GPU）

| 场景 | 参数 | 结果 |
|---|---|---|
| 移动生成微基准 | `make bench` | 1963 万次/秒（单线程）；随机对局 372 万步/秒；UCT-rollout 7.8 万次模拟/秒 |
| 随机跑谱 | 20 万条，16 worker | 0.43 秒，约 47 万样本/秒 |
| 纯 C++ UCT-rollout 跑谱 | 64 次模拟 | 约 2000 样本/秒量级（受 rollout 长度支配） |
| 评测：随机策略 | 200 局 | 均分 1200，约 11 万局/秒 |
| 评测：UCT-rollout | 64 次模拟，200 局 | 均分 15169，43/200 达到 2048，11 秒 |
| NN 自对弈（随机初始化模型） | 64 次模拟，32 lane，in-flight 4 | 1160 样本/秒，平均每批 439 行，叶子缓存命中 60% |
| 管线时间分布（同上） | | `evaluate`（GPU 前向）93%，`advance`（C++ 搜索）5%，`submit` 2% |

### 训练与评测（本轮实测）

| 场景 | 参数 | 结果 |
|---|---|---|
| 价值头训练 | 100 万条，batch 8192，fp32 | 约 22 万样本/秒（每 epoch 2.4 s / 95 万条） |
| 同上，bf16 | `--amp bf16` | 约 23 万样本/秒（**只 +5%**，不足以换精度风险） |
| 同上，`channels_last` | 开/关 | 无差别（±2%） |
| 同上，`torch.compile` | `--compile` | **慢 28 倍**（单 epoch 35 s）——小网络上是负收益 |
| 纯网络评测（单次前向） | nn_value / nn_policy | 57 / 97 局每秒 |
| NN-MCTS-64 评测 | 64 次模拟，64 lane | 2.5 局/秒（每局约 808 步） |

训练循环分阶段计时（batch 8192，单位 ms/批）：

| 阶段 | 耗时 | 说明 |
|---|---|---|
| 数据解码 | 1.1 | 结构化视图 + 批量转张量，已向量化 |
| H2D 传输 | 0.9 | 每批约 0.6 MB |
| D4 增强 | 4.1 | 在 GPU 上做；`--no-augment` 实测不改变总吞吐 |
| **前向** | **17.9** | 4×4 空间上 3×3 卷积利用率低，是当前主要成本 |
| **反向** | **20.8** | 同上 |

结论与行动：

* **GPU 计算占每批的 82%**，所以“把数据管道再优化一点”不是重点；
  真正的杠杆是网络在 4×4 空间上的卷积效率（例如把深层换成 1×1 卷积或 MLP，
  或减少通道数）——那属于架构决策，需要单独的对照实验，不在这轮范围内。
* **每个进程有约 7 秒固定开销**：`import torch` 0.8 s，而 `torch.optim.AdamW` 会触发
  `torch._dynamo` 的惰性导入约 6 s（torch 2.14 的行为，与是否使用 `torch.compile` 无关）。
  短跑（几百步）会显著被它摊薄，因此**优先把一次运行跑长**，而不是频繁起停。
* 自对弈/评测侧的批大小才是吞吐关键：`lanes` 与 `in_flight_per_lane` 扫描结果见下。

### 搜索健康度与噪声底噪（做架构对照前的量尺）

```bash
# 根访问分布诊断：直接读数据里的 visits 与 legal_mask 列
uv run python - <<'EOF'
from r2048.data.shards import BatchReader
import numpy as np
reader = BatchReader("$R2048_OUTPUT_DIR/data/<批>")
rec = reader.gather(np.arange(min(200000, reader.samples)))
visits = rec["visits"].astype(np.int64)
legal = np.stack([(rec["legal_mask"].astype(np.int64) >> a) & 1 for a in range(4)], axis=1)
multi = legal.sum(1) >= 2
print("只访问一条根边:", float(((visits > 0).sum(1)[multi] == 1).mean()) * 100, "%")
print("平均访问边数:", float((visits > 0).sum(1)[multi].mean()))
print("根访问和 == 模拟次数:", float((visits.sum(1) == rec["simulations"]).mean()) * 100, "%")
print("非法方向访问数:", int((visits * (1 - legal)).sum()))
EOF
```

实测：只访问一条根边 **0.000%**，平均访问 3.5 条边（旧版修复前是 99.687%）；
根访问数之和 = 模拟次数 − 1（那一次用于求值根节点本身）。

**噪声底噪**（同配方同数据、只换训练种子、评测 400 局）：训练 1σ ≈ 10~18 评分点，
400 局评测 1σ ≈ 12 点 → **单 seed 对照的可辨识下限约 40~50 分**。
架构对照因此需要参数匹配 + 同数据同初始化 + ≥3 个 seed。

几条与设计直接相关的读数：

* **每批前向的行数**是这套两阶段协议的生命线。早期实现每个 lane 只允许一条模拟在途，
  批次小到 4 行、吞吐 13 样本/秒；允许在途多条之后每批 400+ 行，吞吐提升约 90 倍。
* **C++ 搜索只占 5%**：这条读数是“不给 tree 内部加并行”的依据（见 ADR 0003）。
* 叶子缓存命中率约 40%~60%，直接减少的是跨语言往返次数。
* **自对弈批处理参数扫描**（64 次模拟，2 万样本，随机初始化模型）：

  | lanes | in_flight | 样本/秒 | 每批行数 | 叶子缓存命中 |
  |---|---|---|---|---|
  | 32 | 4 | 1238 | 424 | 58% |
  | 64 | 4 | 1685 | 784 | 58% |
  | **64** | **8** | **1943** | **1762** | 40% |
  | 96 | 8 | 1734 | 2049 | 53% |

  `64/8` 是当前默认；96 lane 反而下降（批更大但缓存命中与调度收益变差）。
  注意 `in_flight_per_lane` 会改变搜索树上并行推进的模拟数，属于**数据生成协议的一部分**，
  因此它写进了 manifest（换值就换了数据的语义）。

## 常见问题

**`make build` 报找不到 pybind11**
`make build` 会自己传 `-Dpybind11_DIR=...`；如果手工调 cmake，需要：
`cmake -S . -B build -Dpybind11_DIR=$(python -m pybind11 --cmakedir)`。
只想要 CLI 与测试时用 `-DR2048_BUILD_PYEXT=OFF`。

**Python 侧 `import r2048_core` 失败**
扩展在 `build/pyext/` 里，需要 `PYTHONPATH` 指向它（`make py`/`make test-py` 已代劳）。
检查 `ls $HOME/r2048/build/pyext/` 是否有 `r2048_core.cpython-312-*.so`。

**构建身份（build id）为何落后于最新提交**
它由 `configure_file` 生成（内容不变时不更新时间戳，避免无谓重编），反映的是最近一次
configure 时的仓库状态。`make build` 每次都会先 configure，所以正常使用不会落后。

**`verify` 的重放校验很慢**
它会把整批数据重新生成一遍，成本与生成相同。只想做快速结构检查时加 `--no-replay`。

**数据批目录已存在**
`generate` 拒绝覆盖：跑谱批按“整批要么完整、要么重来”管理。换个 `--out` 或先清理。
想减少重来代价就把 `--samples` 切小、多跑几批。

**改动跨语言契约（动作顺序、棋盘含义、记录格式）**
必须同时抬高 `cpp/core/version.h` 里的版本号：数据与 checkpoint 都靠它判断语义是否一致，
不匹配会直接失败而不是猜测。
