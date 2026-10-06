#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace r2048 {

struct EvaluateConfig {
    std::string policy = "mcts";
    uint32_t simulations = 0;
    uint64_t games = 0;
    uint64_t seed = 0;
    uint32_t workers = 0;  // 0 = 自动取硬件并发数
    // 评测默认 0：直接取访问数最大的方向，不引入采样噪声，保证同一策略同一开局结果可复现。
    uint32_t temperature_moves = 0;
    std::string jsonl_path;  // 为空则不写逐局记录
    std::string label;       // 写入逐局记录的策略标签
};

struct GameOutcome {
    uint32_t episode = 0;
    uint64_t score = 0;
    int max_exponent = 0;
    uint32_t steps = 0;
};

struct EvaluateResult {
    std::vector<GameOutcome> games;
    double mean_score = 0.0;
    double median_score = 0.0;
    int max_exponent = 0;
    std::map<int, uint32_t> tile_counts;  // 指数 -> 达到该块的局数（含更高块）
    double seconds = 0.0;
};

// 跑固定局数的对局并汇总成绩。
//
// 评测与跑谱共用同一套开局（局号 -> 开局棋盘），因此不同策略、不同模拟次数之间的
// 对比是配对的：第 k 局在所有配置下都从同一局面开始。
EvaluateResult run_evaluate(const EvaluateConfig& config);

}  // namespace r2048
