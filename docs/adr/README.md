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
| [0009](0009-dataset-and-split.md) | 训练数据的读取与划分口径 | 已采纳 |
| [0010](0010-training-recipes-and-gates.md) | 训练配方与门槛 | 已采纳 |
| [0011](0011-evaluation-protocol.md) | 评测口径与比较方法 | 已采纳 |
| [0012](0012-rating-scale-and-iteration-experiment.md) | 评分标度、配对诊断与两轮迭代实验 | 已采纳 |
| [0013](0013-iteration-results.md) | 两轮迭代实验结果：朴素自举会把价值头训坏 | 已采纳 |
| [0014](0014-search-health-and-noise-floor.md) | 搜索健康度诊断与噪声底噪 | 已采纳 |
| [0015](0015-candidate-level-supervision.md) | 候选级监督 | 已采纳 |
| [0016](0016-protocol-variables-and-simulation-budget.md) | 评测协议的隐蔽变量、模拟预算与探索系数 | 已采纳（决策 5/6 已实施，归因已订正；缺口 A/C 未补） |
| [0017](0017-batch-scheduling-dependence.md) | 批组成不是浮点问题：在途模拟的调度依赖与强度代价 | 已采纳 |
| [0018](0018-protocol-variable-registry.md) | 协议变量收口：清单、归属与拒绝兜底 | 已采纳 |
| [0019](0019-jobs-and-progress.md) | 作业目录、结构化进度与子进程封装 | 已采纳（C++ 子命令进度已结构化，见 0020 决策 10） |
| [0020](0020-selfplay-lifecycle-and-incremental-commit.md) | 跑谱生命周期与增量提交（不限量 / 暂停 / 继续 / 终止） | 已采纳 |
| [0021](0021-multi-source-identity-and-training-config.md) | 多来源数据身份与训练配置（YAML） | 已采纳 |
| [0022](0022-training-resume-checkpoints.md) | 训练断点续跑（批级 checkpoint 与随机数状态） | 已采纳 |
| [0023](0023-iteration-orchestration.md) | 迭代编排（薄驱动、步骤标记与半模型合并） | 已采纳 |
| [0024](0024-model-definition-registry.md) | 模型定义的注册表与版本 ID | 已采纳（决策 2 的论证已由 0026 订正） |
| [0025](0025-inference-precision.md) | 推理精度的现状与暂不引入降精度开关 | 已采纳 |
| [0026](0026-model-naming-and-tiers.md) | 模型命名分层与版本层级（测试版 / 正式版） | 已采纳 |
| [0027](0027-inference-pipeline-utilisation.md) | 推理管线的性能诊断与引擎优化方向 | 已采纳 |
| [0028](0028-multi-threaded-search.md) | 多线程搜索（K 个引擎共享写入侧 + 合并推理线程） | 已采纳（决策 2 的前提已被 0031 的实测否掉：K>1 时内容不可复现） |
| [0029](0029-one-batch-many-writers.md) | 一个数据批、多个写者（跨进程并发跑谱） | 已采纳 |
| [0030](0030-value-label-ordering-collapse.md) | MCTS 退化：价值标签的同根排序塌缩；改续弈策略被证实（退化不复现，深度不是杠杆） | 已采纳（三版处方被实测否掉，第四版被证实） |
| [0031](0031-why-parallel-selfplay-is-not-reproducible.md) | 并行跑谱不可复现的原因：挂起不回填（不是缓存） | 已采纳 |
| [0032](0032-advantage-weighted-policy-improvement.md) | 策略改进算子（Â 加权一步改进）首轮实测：不改善、强推变差；归一化尺度在弱信号处反而推得更狠 | 已采纳（负结果；默认仍是 visits） |
| [0033](0033-continuous-selfplay-monitoring.md) | 连续自举五轮的监控：没有劣化、仍未收敛，进步全在价值头而非策略头 | 已采纳 |
| [0034](0034-a-versus-r-plus-a-ranking.md) | 候选排序的两套口径：引擎用 r + A、训练与离线指标用 A；两者之差必须报出来 | 已采纳（换挑选口径实测为空实验：权重逐位相同；真正会变的是 Q 空间的排序项） |
| [0035](0035-q-space-pairwise-ranking-loss.md) | Q 空间的同根成对排序项：贪心判决 +19%（显著），256 模拟的搜索不动 | 已采纳（机制在瞄准的口径上生效；默认关闭） |

新增 ADR 时复制 [template.md](template.md)，编号递增，不要覆盖历史记录：
即使后来的决定推翻了前面的，也只把前面的状态标成“已废弃（被 NNNN 取代）”。
