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

已完成：构建体系、规则引擎、查表移动生成、纯 C++ 跑谱（随机 / UCT-rollout）、
数据格式与校验、批量 NN-MCTS 引擎、Python 批推理与自对弈驱动、双头网络与编码。

尚未实现（下一步）：训练器与断点续跑、模型注册与发布、评测评分池、机器画像。
接口与数据格式已经为这些留好位置（见 ADR 0007、0008）。

实测性能读数见 [docs/BUILD.md](docs/BUILD.md) 的“性能记录”一节。
