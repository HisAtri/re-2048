# ADR 0047：双塔正式代号与本地模型发布

- 状态：已采纳
- 日期：2026-10-10

> 更新：自动发号、迭代发布及数据规模继承由 [ADR 0048](0048-automatic-identities-and-data-lineage.md)
> 订正。当前文件名和清单规模均使用累计来源去重口径，manifest 版本为 2。

## 背景

训练 checkpoint 适合继续训练，却没有独立的发布身份清单；单靠文件名无法校验权重和追溯数据。
ADR 0026 的发布层此前没有实现，ADR 0024 的 safetensors 转换也处于待办状态。

## 决策

1. 双塔网络正式代号改为 `smnet-b9c64dt`：总计 9 个残差块、64 通道、双塔，默认拓扑仍为
   共享 3 块 + 策略塔 3 块 + 价值塔 3 块。`dt-s3p3v3c64`、`dual_tower_v1`、`20261006001`
   保留为历史别名；新训练、合并和发布产物使用 canonical。名字只用于注册表查找。
2. 本地发布统一调用 `python -m r2048 publish`，默认仓库为 `output/models/`（遵循输出根覆盖）。
   每次发布一个独立目录，目录名和模型身份均为 `<架构代号>-<发布日期+流水号>-s<数据集样本数>`。
   文件名约定保留三段结构，扩展名固定为用户指定的单数 `.safetensor`，例如
   `smnet-b9c64dt-20261007001-s1000000.safetensor`，内容使用标准 safetensors 格式。
3. 正式发布要求双头完整模型（`dual` / `joint`），尺寸必须与注册代号的默认规格一致。
   训练最佳权重或两半合并件作为输入；转换仅保留模型张量及 JSON 编码的完整 metadata，
   不包含优化器。训练断点仍用 `.pt`；发布模型可用于推理、评测、自对弈与下一轮训练初始化。
4. 自动日期使用 UTC，与 `published_at` 一致；当天流水号从已完成发布清单取最大值加一。
   可显式指定有效 `YYYYMMDDNNN`，流水号 001–999。已有发布目录拒绝覆盖，历史发布不重写。
   日期流水号是发布批次标签，实际提交时间以 `published_at` 为准。当前编号在跨进程文件锁下
   预留，预留编号不回收；此项由 ADR 0048 更新。
5. 同目录 `manifest.json` 包含 `manifest_version`、`model_name` / `model_id`、`source_model_id`、
   `architecture`、`architecture_params`、`model_file`、`model_sha256`、`release_id`、`published_at`、
   `dataset_samples`、`dataset_ids`、逐数据集规模 `datasets`、`parent_models`、权重精度与训练目标。
   SHA256 针对发布模型文件完整字节计算；父模型记录 ID 和已有的源 checkpoint SHA256，
   从头训练时为空列表。允许两半有不同父模型，因此父模型采用列表，避免丢失一条谱系。
6. 数据规模现指祖先与本轮的累计来源去重覆盖样本数（ADR 0048），按稳定 `source_key` 合并，
   包含验证划分，不乘 epoch 或采样权重。同一来源的训练 limit 是前缀，继承和两半合并时取最大
   覆盖数；零权重来源不计。批次 ID 用于展示；历史没有 batch_id 时保留稳定 key。旧合并件可从
   明确的半模型文件与训练报告补齐，无法验证的谱系阻止发布，不从文件名猜来源或累计规模。
7. 在发布根的临时目录转换，重载并逐张量校验与源权重相等，写 SHA256 和 manifest，最后
   原子重命名整个目录提交。临时目录不进入 WebUI 清单。只有转换和清单同时完成才算发布成功。
   发布 metadata 的 `model_id` 使用正式发布名，原训练 ID 保存为 `source_model_id`。

## 理由

safetensors 让发布产物无须 pickle，JSON 内嵌 metadata 保持嵌套参数与来源信息完整。
外部 manifest 便于仓库索引和文件校验；模型内的 metadata 继续承担独立加载语义，重命名文件
不会改变网络解释。用目录提交避免发布中断留下权重和清单不匹配的可见产物。

## 影响

本 ADR 更新 ADR 0026 的双塔 canonical 与发布扩展名，落实其发布命名待办及 ADR 0024 的
safetensors 待办。历史文件不改写；旧代号仍可加载。没有新的量化或精度转换路径。
