# ADR 0024：模型定义的注册表与版本 ID

- 状态：已采纳
- 日期：2026-10-07

> safetensors 发布与加载待办已由 [ADR 0047](0047-local-model-publication.md) 落实。

## 背景

架构迭代的路线已经定下来了：通道数与深度要改、价值塔与策略塔要往非对称走、
后面还要加 Self-Attention，总体方向是"共享编码器 + 任务专属塔"。这要求模型定义能被
**频繁替换**，而其他代码（数据、跑谱、训练、评测、评分、发布）不必跟着动。

动手之前先量了一下耦合面：除 `nn/models/` 之外，全项目对模型的接触只有 `state_dict()`
（不透明）；C++ 侧完全不知道网络结构（`grep -rn "channels\|architecture\|resnet" cpp/` 为空）；
自对弈、评测、评分、半模型合并都只经一个运行时契约 ——
`model(encoded) -> (value, logits)`。也就是说"不动输入与双头输出就不用动别的代码"
基本成立，剩下的只是几条**还没写清楚的接缝**：

1. 模型类、构造分派、超参清单散在同一个文件里，加一个版本要同时改三处；
2. `architecture_params` 在两个地方各手写一份（类的 `metadata()` 与 `create_model` 的
   每个分支），漏一处就会用默认值**静默重建出另一个网络**（旧版 `create_model()` 忽略
   `shared_blocks/tower_blocks`、永远按 3/3 建网，就是这个坑，写在它自己的注释里）；
3. `head_width` 已经在构造函数里，却没进 `architecture_params` —— 同一个洞，换个参数就重演；
4. **没有从配置造非默认尺寸模型的通路**（`create_model({"architecture": name})` 只拿得到一个
   名字），于是"改通道数/改层数"必须改代码；
5. 没有一个测试会在新增定义时自动检查"能否用自己的 metadata 重建"。

## 决策

1. **一版一个模块**（`nn/models/<name>.py`），注册表是 `nn/models/__init__.py` 里的
   `REGISTRY`：**版本 ID → 定义类**。加一个新版本 = 新增一个模块 + 在注册表加一行；
   改宽度/深度**不需要**动注册表。

2. **ID 是索引，不是规格**。它只回答"这份权重要用哪个定义构造"，**不用来反推结构或超参**：
   超参一律从 metadata 的 `architecture_params` 读；"改宽度"因此是**纯配置行为**，
   只有出现新的拓扑形状才新增定义。

   > **订正（2026-10-07，见 [ADR 0026](0026-model-naming-and-tiers.md)）**：本决策原先附带的
   > 论证有两处错误——① 说"把结构写进 ID 会逼着 ID 随宽度调整改名、于是各处都要动"，
   > 但**换宽度本来就是新模型、就该有新名字**，改名不是代价；② 说旧版的 `s3p3v3-c64`
   > "只能靠 ID 反推结构"，这对旧版是描述错的（它是从 metadata 算出代号，毛病在
   > `create_model` 忽略 `shared_blocks/tower_blocks`）。真正要守的界线只有一条：
   > **名字不参与结构推导**，而不是"名字不该有信息量"。名字该怎么起、测试版与正式版
   > 如何分层，见 ADR 0026。

3. **超参只写一次**：写在 `__init__` 签名里（都有默认值）并**存成同名实例属性**，
   由 `contract.default_params()` / `instance_params()` 读出来，`metadata()` 用
   `definition_metadata()` 生成。于是"加一个超参"= 改签名 + 存属性，metadata 与重建自动跟上；
   漏存属性会在构造时报 `AttributeError`，不会等到某天用默认值静默重建。

4. **加载闸门新增"版本 ID 必须存在且受支持"**：读者（`load_checkpoint`）遇到不认识的 ID
   就退出，消息里列出本版本支持的定义；**缺失 ID 同样算失败**（没有 ID 就无法确认该用哪个
   定义解释这份权重）。理由与 ADR 0018 的"缺字段必须报错"一致：用一个不认识的定义去解释
   权重，最坏情况不是抛异常，而是 `load_state_dict` 恰好成功（形状撞上），得到一个
   "能跑但语义全错"的模型。随机初始化的自检路径（没有 metadata）仍走默认定义。

5. **配置 → 模型通路**：`TrainConfig.architecture`（ID）+ `TrainConfig.architecture_params`
   （超参），YAML 的 `train:` 小节同名可写。同时**拒绝**"给了 `init_checkpoint` 又指定架构/
   超参"的组合：那时结构由 checkpoint 决定，命令行里的设置会被静默忽略——而"我明明改了宽度"
   是最难查的一类困惑。

6. **定义拥有自己的内部实现**：`forward(encoded) -> (value, logits)` 是唯一边界，
   输入是 `encode_boards` 的输出（`(N, 25, 4, 4)`），输出是 `(N,)` 与 `(N, 4)`。
   **位置编码、注意力、塔的非对称都在定义内部实现**，外部不会多传一个参数。
   位置编码尤其重要：对 16 个格子做 self-attention 时集合本身是置换等变的，
   不加位置信息就是错的——而那属于定义内部的事（学一个位置嵌入即可，输入编码不用动）。

7. **已发布的定义不允许删除或改写**：旧权重、评分记录、历史对照都依赖它仍然可加载。
   需要不同语义就新增一个版本 ID（这也是"一版一模块"的用处）。现有的两个 ID
   （`dual_head_resnet_v1` / `dual_tower_v1`）已被 20 个 checkpoint 记住，因此**只增不改**。

8. **`nn/model.py` 退化成加载/构造/冻结这一层**（`create_model` / `load_checkpoint` /
   `save_checkpoint` / `freeze_value_path` / `encode_and_forward`），并转出注册表与结构名
   常量，于是既有 import 路径不变、调用方零改动。

## 理由

* **为什么 ID 不编码结构**：见决策 2。一句话：结构写进 ID 会让每次调宽度都变成一次
  "跨文件改名"，而它换来的信息（"这具权重多宽"）本来就该从 metadata 读。
* **为什么超参要自动记**：这个坑本项目的旧版已经踩过并写在注释里；靠"记得同步两处清单"
  是维持不住的，只有让 metadata 成为**读实例属性**的结果，才在结构上排除了漏记。
* **为什么未知 ID 要退出而不是猜**：形状碰巧相同的两个定义会让 `load_state_dict` 成功，
  于是错误从"加载失败"变成"结果莫名其妙变差"——后者要花掉一整轮实验才可能发现。
* **为什么位置编码放在定义内部**：它是网络结构的一部分（卷积靠局部性隐含位置，注意力
  必须显式加），而不是输入格式的一部分；放进定义就保住了 `INPUT_ENCODING` 不变、
  外部传参不变，也保住了"新旧模型能在同一把尺子上比较"。

## 替代方案

* **把结构规格写进 ID**（旧版 `s3p3v3-c64`）：见上。
* **用 JSON/YAML 描述任意拓扑（把网络做成配置驱动的通用架构）**：灵活，但这等于自己写一套
  框架，而本项目只在几个点上变（宽度、深度、塔形状、后面的注意力）。一版一个模块的
  Python 代码可读、可断点、可 diff，比自造 DSL 便宜得多。
* **保持单文件、只把 `create_model` 改成查表**：能解决分派，但解决不了"超参两处手写"与
  "改宽度要动代码"；而这两条正是接下来最频繁的改动。
* **让 `load_checkpoint` 在未知 ID 时回落到默认定义并警告**：形状对不上会在 `load_state_dict`
  报错（响的），但形状恰好相同时会静默错——一个"警告"配不上这种后果。
* **给 `nn/models/` 加一个自动发现（扫目录/入口点）的机制**：注册表只有几行，显式列出
  本身就是"本版本支持哪些定义"的声明；自动发现会让"支持什么"变成运行时才知道的事。

## 验证方式

* **`python/tests/test_model_registry.py`（6 条）**：
  * `test_every_definition_round_trips` **按注册表参数化**：每个在册定义都做
    "构造 → metadata → 重建 → 前向 → 能用旧权重 strict 加载"，且价值量纲是归一化的
    （< 100）。这条测试对**将来新增的定义自动生效**，是"加架构只动 nn/models/"的守卫。
  * `test_hyperparameters_are_recorded_not_inferred`：每个定义的签名超参都有默认值、
    `head_width` 与 `groups`（默认 8）都在参数表里、每个超参都存成了同名属性；
    非默认尺寸（`channels=32/head_width=64`）能往返，且**换尺寸后旧权重装不上**（响的）。
  * `test_unknown_definition_is_refused`：不认识的 ID 在构造与加载两处都报错、消息里带上
    被拒的 ID 与本版本支持的全部 ID；**缺失 ID 也失败**。
  * `test_registry_self_check_guards_copy_paste`：类与模块声明的 ID 必须一致、
    `__init__` 不许出现 `**kwargs`（那会让超参记不全）。
  * `test_architecture_params_reach_the_model_from_config`：YAML →
    `TrainConfig.architecture_params` → 模型尺寸真的变了，且没提到的字段回落默认值。
* **向后兼容（实测）**：`$R2048_OUTPUT_DIR/runs/*/model.pt` 的 **20 个既有 checkpoint
  全部 strict 加载通过**（它们记录的 `architecture_params` 只有 `channels/blocks`，
  说明旧记录的 `head_width`/`groups` 是靠默认值补齐的，而默认值与原实现一致：
  `head_width=256`、`groups=8`，实测权重形状 `value_head.1.weight = (256, 1024)`）。
* **尚未验证 / 待办**：
  1. **策略路径仍按参数名前缀 `policy` 识别**（`freeze_value_path` 与 `merge_halves`）。
     现有两版都在顶层命名 `policy_tower` / `policy_head` 所以没问题，但把策略塔嵌进父模块
     （`towers.policy.*`）或让策略路径包含一部分主干时，前缀就不再匹配（会**报错**而非静默）。
     这是本 ADR 留下的唯一一处"外部要知道定义内部命名"的耦合，替换方案是"由定义声明策略
     路径"；等第一个非对称塔的定义落地时一起做。
  2. `.safetensors` 未做（用户确认今天不做转换）。**为它预留的一条约束**：safetensors 的
     metadata 头只接受 `str -> str`，所以届时嵌套的 `architecture_params` 需要 JSON 序列化
     成一个字符串；保持"metadata 的值都是可 JSON 化的标量/容器"这条口径，转换时就只是编码问题。
  3. 新架构的**性能与批组成依赖要重新实测**（ADR 0003/0017 的依赖幅度与 `lanes`/`in_flight`
     的最优值都是结构相关的）："模型迭代不影响其他代码"成立，"不影响其他结论"不成立。
