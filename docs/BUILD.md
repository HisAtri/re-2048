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

### 候选级监督（推荐路线，见 ADR 0015）

搜索产出的数据自带候选块，训练时价值头默认训练在**搜索实际查询过的全部合法候选**上：

```bash
# 搜索产出的批次都带候选（纯 MCTS 与 NN-MCTS 都有）；策略目标取自同一批数据的根访问分布
make py ARGS="train --data $R2048_OUTPUT_DIR/data/<搜索批> --output $R2048_OUTPUT_DIR/runs/value-cand --mode value --epochs 8"
# 价值头与策略头一起训（joint / dual），价值用候选目标、策略用访问分布，一次前向覆盖两种行
make py ARGS="train --data <搜索批> --output <run> --mode dual --epochs 8"
# 只要主轨迹口径（复现旧结果或数据没有候选时）
make py ARGS="train --data <批> --output <run> --mode value --value-target trajectory"
```

训练日志里价值头的指标是**候选 MAE / 同根成对排序一致率 / 最优候选命中率 / 后悔值**，
checkpoint 的挑选也按“最优候选命中率”而不是全局 MAE——全局 MAE 无法替代同根候选排序。
数据里没有候选（随机策略批次）时自动退回主轨迹 MC 回报口径。

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

### 跑谱的生命周期（不限量 / 暂停 / 继续 / 终止）

跑谱不再是"给多少局就跑多少局、中途不停"的形状（见 ADR 0020）：

```bash
# 不限量：跑到被叫停为止（--samples 0 是默认值）
make py ARGS="selfplay --checkpoint runs/x/model.pt --simulations 64 --shard-samples 20000"

# 停车：把 pause 或 stop 写进控制文件（默认 <数据批目录>/control），一次性消费
echo stop  > $R2048_OUTPUT_DIR/data/nn-s64-20261007-120000/control
echo pause > $R2048_OUTPUT_DIR/data/nn-s64-20261007-120000/control

# 接着跑：同样的参数加 --continue，数据写进同一个批次目录
make py ARGS="selfplay --checkpoint runs/x/model.pt --simulations 64 --shard-samples 20000 \
  --samples 200000 --continue --out $R2048_OUTPUT_DIR/data/nn-s64-20261007-120000"
```

几条要点：

* **停车点只有"局与局之间"**：进行中的局会走完，已产出的数据一定被提交。
  一次 Ctrl-C 等价于 `stop`（再按一次才硬退出）。`kill -9` 也只丢掉"尚未落盘的那一屏"，
  已提交的分片与清单都是完好的——清单是**增量提交日志**，它列出的分片才有效。
* **续跑参数逐项比对**：`seed / policy / simulations / temperature_moves / model_id /
  build_id / shard_samples` 与协议参数（`c_puct / lanes / in_flight_per_lane /
  temperature / cache_entries / max_batch / device`）必须一致，不一致会**指名**拒绝
  （"seed 原为 888，本次为 999"）。非协议项（`--samples`、`--workers`、`--out`、`--control`）
  可以改。跑满目标的批次（`state: complete`）不允许追加，请新起一批。
* **`--samples` 在续跑时是整批的累计目标**：想"再要 5000 条"就填当前总数 + 5000。
* **逐字节复现的边界**：纯 C++ 跑谱（`generate`，整数运算）暂停+续跑与一次跑完
  **逐字节相同**（`make test-cpp` 里有对应单测）；NN 自对弈的叶值依赖批组成（ADR 0017），
  续跑进程的批组成与前一次天生不同，所以**前缀逐字节相同、尾部可能不同**
  （实测 6000 条目标下 54 局里 10 局）。同一个停车点续跑两次是逐字节相同的。

## Python 侧

```
python -m r2048 version          # 两侧版本、构建身份、torch/CUDA 状态
python -m r2048 env              # 输出根等路径
python -m r2048 selfplay ...      # NN 引导的批量自对弈（需要 --checkpoint，或用 --random-init 自检）
```

扩展模块 `r2048_core` 由 CMake 构建到 `build/pyext/`，通过 `PYTHONPATH` 引入
（Makefile 已代劳）。它不在 PyPI 上，也不会被装进 site-packages——源码树里因此
不会出现任何二进制产物。

### 训练配置（YAML）

训练是唯一带配置文件的环节（ADR 0021）：它的参数属于"一条训练线"，要被反复引用、
被记进 checkpoint、被迭代编排复用。跑谱 / 评测 / 评分仍然只有 CLI。

```yaml
# train.yaml
datasets: # 用它训练哪些数据，多来源在这里列
  - path: $R2048_OUTPUT_DIR/data/teacher-s256
    weight: 1.0 # 每轮采样的相对配额；0 = 不用这个来源
  - path: $R2048_OUTPUT_DIR/data/teacher-s1024
    weight: 0.25 # 小比例混入
    limit: 0 # 可选：只用该来源的前 N 条（0 = 不限）
train:
  mode: value
  epochs: 8
  batch_size: 4096
  lr: 3.0e-4
  seed: 11
```

```bash
make py ARGS="train --config train.yaml --output $R2048_OUTPUT_DIR/runs/value-w1"
# --data 是单批用法，给了它就以它为准（编排用它把训练线切到刚跑出来的数据上）
make py ARGS="train --config train.yaml --data $R2048_OUTPUT_DIR/data/new --output ..."
```

几条要点：

* **优先级 CLI > YAML > 默认值**，`train:` 里的字段名与 `TrainConfig` 一致；
  出现未知小节 / 字段 / 数据键都直接报错（拼错的键被忽略等于"改了半天没生效"）。
* **来源身份 = `(source_key, episode)`**：局号只在批内有效，`source_key` 由 provenance
  （策略、模拟数、seed、模型、build、协议参数、schema/rules/protocol 版本）算出，
  与目录名、批次号无关。合并多个来源时**每个来源自己的划分不会变**，
  而"同 seed 的同一局号"不再被判成同一局。
* **权重是每轮配额**（`quota_i ∝ w_i × 训练池_i`，每轮长度默认 = 各来源训练池之和）。
  **权重 0 等于没列这个来源**：训练曲线与"不列它"逐张量相同。
* 训练日志逐来源报出训练/验证局数；`train.json` 与 checkpoint 元数据逐来源记身份与权重；
  **验证集为空直接报错**（除非显式 `val_fraction=0`）。

### 训练断点续跑

训练默认每约 10 分钟（或每 `--save-every-steps N` 步）在运行目录里原子写一次 `last.pt`，
它停在**批边界**上（见 ADR 0022），因此多数情况下续跑与"一次跑完"逐张量相同：

```bash
make py ARGS="train --config train.yaml --output $R2048_OUTPUT_DIR/runs/value-w1 --save-every-steps 200"
# 崩了、或者机器要还回去，直接接着跑（同一个 --output）
make py ARGS="train --config train.yaml --output $R2048_OUTPUT_DIR/runs/value-w1 --resume"
```

几条要点：

* **停在批边界、停在块边界**：`last.pt` 记录 `epoch` + `step_in_epoch` + 随机数状态
  （torch / CUDA / numpy / D4 增强发生器），续跑跳过前 `step_in_epoch` 批（不重训、
  也不增强——增强会消耗随机数）。判据是逐张量相等，实测 A 一次跑完 4 轮与
  "第 2 轮第 3 批被杀 → `--resume`"的最终权重 `torch.equal`。
* **数据身份变了会被拒绝**（来源身份 `source_key` / 权重 / `val_fraction` / `split_seed`）：
  换数据接着训是静默错误，想换就换个 `--output` 新起一条线。批次**原地增长**不算换数据
  （来源身份没变），只会在日志里提示池子大小变了。
* **其他配置变了会逐字段打印**（"`epochs: 8 → 12`"）然后继续——把轮数调长是正常操作。
* **`lr` 以配置为准**：恢复调度器会把基准学习率一起恢复，所以代码按配置重算当前步位置上的
  学习率；同一个 `lr` 时这是恒等运算。
* **暂停的是"计算的粒度"**：被打断那一轮的日志只统计续跑跑过的那一截
  （训练损失与验证指标的 tail 平均），后续各轮与最终权重不受影响。

## 性能记录（本机实测，16 核 WSL2 + 直通 GPU）

| 场景 | 参数 | 结果 |
|---|---|---|
| 移动生成微基准 | `make bench` | 1963 万次/秒（单线程）；随机对局 372 万步/秒；UCT-rollout 7.8 万次模拟/秒 |
| 随机跑谱 | 20 万条，16 worker | 0.43 秒，约 47 万样本/秒 |
| 纯 C++ UCT-rollout 跑谱 | 64 次模拟，16 worker | 1.5 万样本/秒（局/秒 × 平均步数；各级预算见下表） |
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
这是**抽样噪声**；NN 路径还有第二个来源（批组成，见下），跨配置比较前先读 ADR 0016。

### 模拟预算与协议变量（ADR 0016）

“模拟数 → 评分”完整曲线（seed 20262053，锚点 = 纯 MCTS-64 = 1000；纯线 800 局/臂、
NN 线 64/128/256 为 800 局、512/1024 为 400 局）。跑法：
`make run ARGS="evaluate --policy mcts --simulations N --games 800 --seed 20262053 --jsonl <路径> --quiet"`
与 `make py ARGS="eval --checkpoint <ckpt> --track nn_mcts --simulations N --games 800 --seed 20262053 --c-puct 1.5 --lanes 64 --in-flight 4 --jsonl <路径>"`。

| 模拟 | 纯 rollout 评分 | ±95% | 秒/局 | NN 引导评分 | ±95% | 秒/局 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1000.0（锚） | 17.0 | 0.061 | 1050.2 | 17.1 | 0.383 |
| 128 | 1082.4 | 16.8 | 0.123 | 1138.3 | 16.3 | 0.875 |
| 256 | 1137.7 | 16.6 | 0.278 | 1180.6 | 16.7 | 1.804 |
| 512 | 1183.8 | 16.3 | 0.628 | 1209.9 | 19.5 | 3.780 |
| 1024 | 1204.6 | 16.2 | 1.333 | **1236.9** | 19.3 | 8.078 |

* **两条曲线到 1024 都还在涨，没有平台**：每翻倍模拟的边际收益是
  纯 +82.4 / +55.3 / +46.1 / +20.8，NN +88.1 / +42.3 / +29.3 / +27.0；
  成本按约 `模拟数^1.15` 增长（每次翻倍约 2.1~2.3× 秒/局）。
  按“每单位墙钟换多少分”看收益每次翻倍约减半 → **性价比拐点在 128~256，绝对最强在 1024**。
* **等预算下 NN 引导全线优于纯 rollout**（+26~+56 分，五条 95% 区间都排除 0），
  强弱关系等价于**把搜索预算放大一倍**：NN-128（1138.3）≈ 纯-256（1137.7），
  NN-512（1209.9）≈ 纯-1024（1204.6）。优势是否随预算衰减在本次样本下不可辨识。
* 评测/自对弈口径建议从 64 提到 **256**；追最强用 1024。生产 100 万条数据的代价
  （纯线路由，16 线程）：s64 约 1 分钟、s256 约 4 分钟、s1024 约 18 分钟（实测外推）。
* 但**候选标签的量级不随纯 rollout 教师预算变化**（263 / 288 / 282 分）：
  rollout 的值是“均匀随机播放到底”的回报，多模拟只降方差。要抬高价值目标要换教师语义
  （NN 引导搜索当教师 → 420 分）。
* `c_puct` 在 0.75~3.0 内对强度无可辨识影响；**但只把 lanes 从 64 改成 32 就有 +20 分的位移**，
  两者同阶。因此 c_puct 不作为强度调参方向，而必须与批组成一起进协议指纹。
* **在途流水线（`in_flight`）不是纯吞吐旋钮，它是吞吐换强度**：`s32`、`lanes 8`、64 局下
  `in_flight=1` 均分 **12522**（到 2048 的局 9.4%），`in_flight=4` 只有 **8923**（0.0%）；
  而两边的每步有效模拟数都恰好是 32，所以丢的不是预算，是同样 32 次模拟被用得更差。
* 同一 seed 下把 `--games` 从 400 提到 800，**前 400 局逐局不变**（已验证），
  所以池子可以跨局数累积；反之，同一条臂在 400 与 800 局下的评分差 ~11 分，
  就是 400 局规模抽样噪声的现场演示。

实测过的复现性边界（改动前必读）：

| 变化 | 结果 |
|---|---|
| 同参数重复跑（纯 C++ 或 NN） | 逐局一致（400/400、24/24） |
| 纯 C++ `workers` 3 → 16 | **逐字节一致**（无浮点，worker 只是性能参数） |
| NN `lanes` 24 → 6 / 64 → 32 | **逐局全不同**（16 局对照下 1/16 一致，见下） |
| NN `in_flight` 1 → 4（`lanes` 固定） | **逐局全不同**（16/16 → 1/16） |
| NN `cache_entries` 131072 → 1024（`lanes` 固定） | **逐局全不同**（`in_flight=4` 时 0/16；`in_flight=1` 时无影响） |
| `device` cuda → cpu（同一 seed、同一参数） | **逐局几乎全不同**（`in_flight=4` 2/16、`in_flight=1` 也只有 3/16），但**聚合量测不出偏移**（见下） |

机制**不是浮点**——别按"锁 dtype / 上确定性算子"的方向修，实测三项确定性开关都压不掉，
且浮点噪声小四个数量级的 CPU 与 GPU 分叉程度一样。真正的通道是**在途模拟的调度切分**：
一条模拟沿路径先把 `edge_visits` 自增、`edge_q_sum` 要等回填才更新，而它会不会进在途
取决于共享推理缓存是否命中该 (board, kind)。于是 `lanes / in_flight / cache_entries`
共同决定了每步的（选择 → 展开 → 回填）序列；根节点取整数访问数的 argmax，
而被翻转的那一步之后，同一串按 (局号, 步号) 派生的随机块落到另一个棋盘上，余下整局分叉。
细节与判别实验见 [ADR 0017](adr/0017-batch-scheduling-dependence.md)。
所以 NN 评测必须冻结并记入协议的变量是
`lanes / in_flight / max_batch / cache_entries / device`（外加 `simulations / c_puct`）。

**协议变量怎么落地**（[ADR 0018](adr/0018-protocol-variable-registry.md)）：清单只有一份，
在 `python/src/r2048/protocol.py` 的 `PROTOCOL_FIELDS`；评测写出方（`evaluate.py` 与
C++ `r2048 evaluate`）按它声明**全部**字段，评分侧按同一份清单校验并哈希。
两条规则决定了它的行为：**缺字段直接拒收**（不再用 `row.get(k, 默认值)` 兜底——
默认值把“没记录”说成“记录了默认值”，两个真正不同的协议就此静默合并），
**`null` 表示该策略没有这个变量**（随机策略没有 `c_puct`，纯网络 track 没有搜索预算）。
逐局记录里因此平铺着 13 个协议字段 + `protocol_hash` 进池时重算，例如 c2 臂的一行：

```json
{"track":"nn_mcts","rules_version":1,"protocol_version":1,"simulations":64,"c_puct":1.5,
 "temperature":0.0,"temperature_moves":0,"inference_precision":"float32","lanes":64,
 "in_flight":4,"max_batch":8192,"cache_entries":131072,"device":"cuda"}
```

`protocol_version` 承担那些**没有单独入口**的实现常量（纯 rollout 的探索常数、
rollout 策略本身、批调度形式）：它们同样改变逐局结果，改动时必须抬高该版本号。
`eval_seed`、`games`、纯 C++ 线的 `workers` **不属于**协议：
seed 是同一协议下的重复抽样（它只进池子的去重键），`games` 只影响区间宽度
（同 seed 加大局数时前 N 局逐局不变），`workers` 实测逐字节一致。

**跨设备**（同一 seed、同一参数，400 局 × 2）：逐局只有 3/400 局得分相同，但 `ΔL = +0.0053`
（两侧独立 1.96σ 合并 ±0.1161，**区间含 0**）= **+1.2 分（95% CI ±26.7）**，
中位 7112 vs 7100、步数 561 vs 559。逐局 `log2(得分+1)` 相关系数仅 **r = +0.098**，
所以跨设备分叉是**重新抽签、不是系统性偏移**；但 ±26.7 分与我们要判的最小效应
（+20 分）同阶，因此还不能宣称设备中性，要 ±10 分需每设备约 2850 局。
`in_flight=1` 也拿不回跨设备一致（同设备内改 `lanes` 是 16/16，跨设备只剩 3/16），
因为跨设备数值差（`|Δvalue|` 3.6e-5）虽然比同设备改批大小的 5.8e-4 还小，
却已足够被搜索这个高增益放大器放大成整局分叉。

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
`generate` 拒绝覆盖已在跑的一批：续写要用 `--continue`（它会逐项比对参数、
从清单记录的局号接着编号，见 ADR 0020）。想另起一批就换个 `--out`。
不限量跑谱下不需要"切小目标、多跑几批"这种变通：跑到够了就 `stop`，之后随时 `--continue`。

**改动跨语言契约（动作顺序、棋盘含义、记录格式）**
必须同时抬高 `cpp/core/version.h` 里的版本号：数据与 checkpoint 都靠它判断语义是否一致，
不匹配会直接失败而不是猜测。
