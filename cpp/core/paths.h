#pragma once

#include <cstdint>
#include <string>

namespace r2048 {

// 输出根目录约定：与 Python 侧 python/src/r2048/paths.py 保持一致。
// 两边各写一份三行实现，是为了避免为了传一个路径而在跨语言边界上加参数
// —— 环境变量是这里最合适的契约，而不是把路径塞进每条命令。
std::string output_root();

// 数据批目录：<输出根>/data/<批次号>
std::string default_data_dir(const std::string& batch_id);
// 评测输出目录：<输出根>/eval
std::string default_eval_dir();

// 默认批次号：<策略>[-s<模拟次数>]-<日期>-<时刻>，便于在目录名上直接读出这批数据的来源。
std::string make_batch_id(const std::string& policy, uint32_t simulations);

}  // namespace r2048
