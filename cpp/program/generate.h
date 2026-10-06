#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "dataio/shard.h"

namespace r2048 {

struct GenerateConfig {
    std::string data_dir;
    std::string batch_id;
    std::string policy = "random";  // random / mcts
    uint32_t simulations = 0;
    uint64_t samples_target = 0;
    uint64_t seed = 0;
    uint64_t shard_samples = 100000;
    uint32_t workers = 0;  // 0 = 自动取硬件并发数
    uint32_t temperature_moves = 0;
    std::string model_id;  // 非 NN 策略留空
    std::vector<std::pair<std::string, std::string>> params;
};

struct GenerateResult {
    uint64_t samples = 0;
    uint32_t episodes = 0;
    double seconds = 0.0;
    std::string manifest_path;
};

// 并行跑谱。
//
// 结构：N 个 worker 各自持有独立决策器按局号取任务，主线程按**局号顺序**写盘。
// 顺序写盘不是保守，而是刻意保证的不变式：数据文件逐字节只由 (种子, 参数, 目标条数)
// 决定，与 worker 数量无关。这样 verify 才能用“换一个线程数重新生成、比对哈希”
// 的方式校验整批数据，也才不会出现“换台机器重跑就得到不同数据集”的问题。
//
// 条数截断同样只依赖目标：最后一批不足一条局的部分样本会被截断，使样本数恰好等于目标。
GenerateResult run_generate(const GenerateConfig& config);

}  // namespace r2048
