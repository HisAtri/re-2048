# 架构决策记录（ADR）

一次改动如果引入了“以后不方便再改”的东西（构建方式、跨语言契约、数据格式、
搜索形式化、目录约定），就必须在这里留一条记录。代码注释解释局部意图，
ADR 解释为什么整体是这个形状。

| 编号 | 标题 | 状态 |
|---|---|---|
| [0001](0001-build-system-and-layout.md) | 构建系统与工程布局 | 已采纳 |
| [0002](0002-platform-and-toolchain.md) | 平台与工具链 | 已采纳 |
| [0003](0003-cpp-python-boundary.md) | C++/Python 职责边界与两阶段批推理 | 已采纳 |
| [0004](0004-rules-and-action-order.md) | 规则语义与动作顺序契约 | 已采纳 |
| [0005](0005-board-representation-and-tables.md) | 棋盘表示与查表移动生成 | 已采纳 |
| [0006](0006-search-formulation.md) | 搜索形式化：UCT-rollout 与 NN 引导搜索 | 已采纳 |
| [0007](0007-data-format-and-provenance.md) | 训练数据格式与来源校验 | 已采纳 |
| [0008](0008-output-layout-and-naming.md) | 输出目录布局与默认命名 | 已采纳 |

新增 ADR 时复制 [template.md](template.md)，编号递增，不要覆盖历史记录：
即使后来的决定推翻了前面的，也只把前面的状态标成“已废弃（被 NNNN 取代）”。
