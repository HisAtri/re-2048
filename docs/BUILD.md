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

几条与设计直接相关的读数：

* **每批前向的行数**是这套两阶段协议的生命线。早期实现每个 lane 只允许一条模拟在途，
  批次小到 4 行、吞吐 13 样本/秒；允许在途多条之后每批 400+ 行，吞吐提升约 90 倍。
* **C++ 搜索只占 5%**：这条读数是“不给 tree 内部加并行”的依据（见 ADR 0003）。
* 叶子缓存命中率约 47%~60%，直接减少的是跨语言往返次数。

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
