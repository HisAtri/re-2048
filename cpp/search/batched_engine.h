#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/version.h"

namespace r2048 {

// 两阶段批推理协议中，一条请求要问的是哪一头。
enum RequestKind : uint8_t {
    kRequestValue = 0,   // afterstate u → 价值头 A(u)（网络输出已按 score_scale 归一化）
    kRequestPolicy = 1,  // 决策局面 s → 策略头 logits（4 个方向）
};

struct EngineConfig {
    uint32_t lanes = 8;              // 同时在跑的局数
    uint32_t simulations = 64;       // 每步的模拟次数
    // 每个 lane 允许同时在途的模拟数。这是吞吐的关键旋钮：
    // 每次模拟都要先算一个叶子，如果在途只有 1 个，整条流水线就退化成
    // “算一个叶子 → 等一次 GPU 往返 → 再算下一个”，GPU 批次小到几行、CPU 全程等。
    // 允许在途多个之后，一次前向能覆盖 lanes × in_flight 个叶子（再乘上每个
    // 展开节点自带的 5 个局面），CPU 搜索与 GPU 推理才真正重叠起来。
    uint32_t in_flight_per_lane = 8;
    float c_puct = 1.5f;             // PUCT 探索系数
    uint32_t temperature_moves = 30; // 开局多少步按访问分布采样
    float temperature = 1.0f;        // 访问数开 1/T 次幂后采样；0 表示直接取最大
    uint32_t cache_entries = 1 << 17;  // 推理缓存条目数（2 路组相联）
    float score_scale = kScoreScale;
    uint64_t seed = 0;
    uint32_t threads = 0;            // lane 并行的线程数，0 = 硬件并发数
};

struct EngineStats {
    uint64_t requests_issued = 0;     // 累计送出的批推理请求条数
    uint64_t requests_submitted = 0;  // 累计回填的条数（应等于送出的条数）
    uint64_t leaf_evaluations = 0;    // 叶子求值次数
    uint64_t leaf_from_cache = 0;     // 其中完全由缓存满足、无需跨语言调用的次数
    uint64_t simulations = 0;
    uint64_t episodes = 0;
    uint64_t samples = 0;

    // 缓存命中率按“叶子求值无需请求”的比例计算：这是真正省掉的跨语言往返。
    double leaf_cache_rate() const {
        return leaf_evaluations == 0 ? 0.0
                                     : static_cast<double>(leaf_from_cache) /
                                           static_cast<double>(leaf_evaluations);
    }
};

// 批量 NN-MCTS 自对弈引擎：C++ 持有多局并发生成的全部状态，Python 只负责神经网络。
//
// 协议是两阶段的（与旧版一致，是这台机器上被验证过跑得动的形态）：
//     advance(max_requests) → 一批 (request_id, kind, board)
//     Python 一次前向 → submit(ids, values, policy_logits)
// C++ 侧绝不逐样本调用 Python：一次决策节点展开会同时需要 1 个策略头结果（决策局面）
// 与最多 4 个价值头结果（各 afterstate），因此每个前向批次天然覆盖 5 倍以上的局面。
//
// 三个必须守住的性质：
//   * 数据逐字节只由 (种子, 参数, 目标条数) 决定：lane 调度顺序、线程数都不影响写入内容。
//   * 每个 lane 的随机流由 (episode, step) 派生，因此任何一局都能单独重放。
//   * 应当并行的部分在 C++ 内并行：advance() 内部用线程池按 lane 并行推进，
//     调用方释放 GIL 后不会出现“CPU 等 GPU、主线程站着等”的串行形态。
class BatchedEngine {
public:
    struct Advance {
        std::vector<uint64_t> request_ids;
        std::vector<uint8_t> request_kinds;
        std::vector<uint8_t> boards;  // N×16 行优先的指数棋盘
        bool done = false;            // 已产出足够样本，可以收尾
        uint64_t samples_written = 0;
        uint32_t episodes_written = 0;
    };

    BatchedEngine(const EngineConfig& config, std::string data_dir, std::string batch_id,
                  std::string model_id, uint64_t samples_target, uint64_t shard_samples);
    ~BatchedEngine();

    BatchedEngine(const BatchedEngine&) = delete;
    BatchedEngine& operator=(const BatchedEngine&) = delete;

    // 推进所有 lane 直到需要神经网络，返回这一批请求。
    Advance advance(uint32_t max_requests);

    // 回填一批结果：values 为归一化后的价值，policy_logits 为 4 个方向的 logits
    // （按请求种类只取其一；长度均为 count）。
    void submit(const uint64_t* ids, const float* values, const float* policy_logits, size_t count);

    EngineStats stats() const;

    // 诊断用快照：每条 lane 的状态与写入侧进度。卡住时唯一能看清全局的东西，
    // 因此保留成正式接口而不是临时调试代码。
    struct MissingNeed {
        uint32_t leaf = 0;
        uint8_t kind = 0;  // 0 决策节点，1 随机块节点
        bool missing_policy = false;
        uint32_t missing_values = 0;
    };
    struct LaneStatus {
        uint32_t episode = 0;
        uint32_t step = 0;
        uint32_t simulations_done = 0;
        uint32_t pending = 0;
        uint32_t outstanding = 0;
        bool finished = false;
        bool needs_new_episode = false;
        std::vector<MissingNeed> missing;
    };
    struct DebugState {
        std::vector<LaneStatus> lanes;
        uint64_t queued_requests = 0;
        uint64_t in_flight_requests = 0;
        uint32_t next_episode = 0;
        uint32_t next_episode_to_write = 0;
        uint64_t completed_buffered = 0;
        uint64_t samples_produced = 0;
        std::vector<uint32_t> completed_keys;
        uint64_t samples_written = 0;
        uint64_t reissues = 0;  // 补发请求的次数（缓存挤出导致的重新请求）
        bool stop_all = false;
    };
    DebugState debug_state() const;

    // 达到目标后收尾：写完最后一个分片并落 manifest，返回 manifest 路径。
    std::string finish();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace r2048
