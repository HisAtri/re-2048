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
    // 0 表示不限量：一直跑到被控制通道叫停为止（见 core/control.h）。
    uint64_t samples_target = 0;
    uint64_t seed = 0;
    uint64_t shard_samples = 100000;
    uint32_t workers = 0;  // 0 = 自动取硬件并发数
    uint32_t temperature_moves = 0;
    // 记录里放几个树内节点（0 = 不采集，记录 192 字节）。它是**格式参数**：不改变任何一局的
    // 走法，但决定记录字节数，因此进清单、进续跑闸门、进 verify 的重放（ADR 0041）。
    uint8_t tree_nodes = 0;
    // rollout 的续弈策略（仅 policy=mcts 有效）：false = 均匀随机（历史行为），
    // true = 一步启发式。它改变候选价值标签的信息量，属协议参数（见 RolloutMctsConfig）。
    bool greedy_rollout = false;
    std::string model_id;  // 非 NN 策略留空
    std::vector<std::pair<std::string, std::string>> params;
    // 控制文件路径。为空时用 <数据批目录>/control：控制通道与批次放在一起，
    // 谁在跑、怎么叫停，看目录就知道。
    std::string control_path;
    // 续跑：在既有批次上接着编号写下去（参数必须逐项一致，否则拒绝）。
    bool resume = false;
    // 进度以 JSON 行输出到 stderr（供父进程解析，见 r2048.jobs.run_cpp）。
    bool progress_json = false;
};

struct GenerateResult {
    uint64_t samples = 0;
    uint32_t episodes = 0;
    double seconds = 0.0;
    std::string manifest_path;
    // 本次运行结束时这一批的状态：complete 表示跑满目标，paused/stopped 表示被叫停。
    BatchState state = BatchState::kStopped;
};

// 并行跑谱。
//
// 结构：N 个 worker 各自持有独立决策器按局号取任务，主线程按**局号顺序**写盘。
// 顺序写盘不是保守，而是刻意保证的不变式：数据文件逐字节只由 (种子, 参数, 样本流)
// 决定，与 worker 数量无关。这样 verify 才能用“换一个线程数重新生成、比对哈希”
// 的方式校验整批数据，也才不会出现“换台机器重跑就得到不同数据集”的问题。
//
// 条数截断同样只依赖目标：最后一批不足一条局的部分样本会被截断，使样本数恰好等于目标。
// 目标为 0 时不截断、不停止，直到控制通道给出 pause/stop——此时正在进行的局会走完
// （局与局之间是唯一的停车点），然后写入终态清单。
GenerateResult run_generate(const GenerateConfig& config);

}  // namespace r2048
