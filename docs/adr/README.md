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
| [0007](0007-data-format-and-provenance.md) | 训练数据格式与来源校验 | 已采纳（重放只对 C++ 教师成立，见文件内 2026-10-10 订正） |
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
| [0029](0029-one-batch-many-writers.md) | 一个数据批、多个写者（跨进程并发跑谱） | 已采纳（监督失败路径与遥测容错见文件内 2026-10-10 订正） |
| [0030](0030-value-label-ordering-collapse.md) | MCTS 退化：价值标签的同根排序塌缩；改续弈策略被证实（退化不复现，深度不是杠杆） | 已采纳（三版处方被实测否掉，第四版被证实） |
| [0031](0031-why-parallel-selfplay-is-not-reproducible.md) | 并行跑谱不可复现的原因：挂起不回填（不是缓存） | 已采纳 |
| [0032](0032-advantage-weighted-policy-improvement.md) | 策略改进算子（Â 加权一步改进）首轮实测：不改善、强推变差；归一化尺度在弱信号处反而推得更狠 | 已采纳（负结果；默认仍是 visits） |
| [0033](0033-continuous-selfplay-monitoring.md) | 连续自举五轮的监控：没有劣化、仍未收敛，进步全在价值头而非策略头 | 已采纳 |
| [0034](0034-a-versus-r-plus-a-ranking.md) | 候选排序的两套口径：引擎用 r + A、训练与离线指标用 A；**默认挑选口径已改为 r+A** | 已采纳（度量的口径错误已无条件修掉；在 r6 数据上该错误尚未显现：两臂权重逐位相同） |
| [0035](0035-q-space-pairwise-ranking-loss.md) | Q 空间的同根成对排序项：贪心判决 +19%、搜索臂在 8/16/64 模拟上显著变强 | 已采纳（默认开启；初版「256 模拟不转移」已订正为分辨率不足） |
| [0036](0036-record-text-format.md) | 棋谱文本格式：一行一命令、生成块显式化、值域键表、解析与应用分两层 | 已采纳 |
| [0037](0037-engine-arbitrary-root-and-step-drive.md) | 引擎的任意局面与逐步驱动（会话模式）：在批引擎上加 set_root / begin_analysis / report_root / commit_step | 已采纳 |
| [0038](0038-session-process-and-text-protocol.md) | 会话进程与文本协议（三层结构）：一行一命令、空行结束应答、棋谱是唯一事实源 | 已采纳 |
| [0039](0039-webui-structure-and-dependencies.md) | WebUI 结构与依赖归属：Gradio 子包 + 可选依赖组、只跟会话说话、清单层不手输路径 | 已采纳 |
| [0040](0040-session-random-stream-and-terminal-invariants.md) | 会话的两个规则 bug（生成块固定位置/只出 2、终局的棋被丢掉）换来的不变量 | 已采纳 |
| [0041](0041-record-v3-tree-nodes.md) | 记录格式 v3：树内节点块（K 由清单声明），v2 数据归档、不维护兼容 | 已采纳 |
| [0042](0042-tree-node-supervision-results.md) | 树内节点监督首轮实测：真实但微弱（+105，四档一致但只一档显著），贪心臂反而变差 | 已采纳（默认 K 经订正为 2：一致性优先于成本） |
| [0043](0043-tree-node-independent-labels.md) | 树内节点的独立标注（记录 v4）：树内排序标签脱离当前头、共享续弈原语；默认采样由 0044 订正 | 已采纳（机制） |
| [0044](0044-tree-label-ab-and-training-gates.md) | 独立树内标签的训练闸门、单变量 A/B 与默认配方验收 | 已采纳（rollout + top_visits） |
| [0045](0045-webui-workspace-design.md) | Gradio 工作台设计系统、响应式布局与编辑复盘流程 | 已采纳 |
| [0046](0046-project-output-root.md) | 数据集与模型默认归档在项目 output/ | 已采纳 |
| [0047](0047-local-model-publication.md) | 双塔正式代号与本地 safetensor 模型发布清单 | 已采纳 |
| [0048](0048-automatic-identities-and-data-lineage.md) | 自动身份分配、迭代发布与累计数据谱系 | 已采纳 |
| [0049](0049-gradio-experiment-queue.md) | Gradio 实验工作台、持久队列、合作式停止与对局资源占用 | 已采纳 |
| [0050](0050-experiment-workbench-layout.md) | 实验页面独立参数、全宽操作区与任务视图 | 已采纳 |
| [0051](0051-machine-performance-profiles.md) | CLI、迭代与 WebUI 统一加载本机实测性能画像 | 已采纳 |
| [0052](0052-assessment-profiles-and-random-seeds.md) | 固定测评配置、共享基线与自动随机种子 | 已采纳 |
| [0053](0053-value-supervision-d4-augmentation.md) | 补齐候选与树内价值监督的 D4 增强 | 已采纳 |
| [0054](0054-model-owned-input-encoding.md) | 模型定义内部编码与原始棋盘输入契约 | 已采纳 |
| [0055](0055-neural-network-module-layers.md) | 网络定义、复用组件、元数据与注册管理分层 | 已采纳 |

新增 ADR 时复制 [template.md](template.md)，编号递增，不要覆盖历史记录：
即使后来的决定推翻了前面的，也只把前面的状态标成“已废弃（被 NNNN 取代）”。
