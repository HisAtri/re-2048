#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/version.h"
#include "dataio/sample_sink.h"
#include "dataio/shard.h"  // BatchState：收尾时要回报这一批停在什么状态

namespace r2048 {

// 两阶段批推理协议中，一条请求要问的是哪一头。
enum RequestKind : uint8_t {
    kRequestValue = 0,   // afterstate u → 价值头 A(u)（网络输出已按 score_scale 归一化）
    kRequestPolicy = 1,  // 决策局面 s → 策略头 logits（4 个方向）
};

// 每步怎么选方向。评测与自对弈必须共用同一套规则与对局推进，只在“选招”这一处分岔，
// 否则三条评测 track（纯价值贪心 / 纯策略 / NN-MCTS）会因为实现对局逻辑而互相不可比。
enum DecisionMode : uint8_t {
    kDecisionSearch = 0,        // 默认：PUCT 树搜索
    kDecisionValueGreedy = 1,   // 不做树搜索：argmax_a [r(s,a) + A(u_a)]
    kDecisionPolicyGreedy = 2,  // 不做树搜索：argmax_a π(a|s)（屏蔽非法方向）
};

struct EngineConfig {
    uint32_t lanes = 8;         // 同时在跑的局数
    uint32_t simulations = 64;  // 每步的模拟次数
    // 每个 lane 允许同时在途的模拟数。这是吞吐的关键旋钮：
    // 每次模拟都要先算一个叶子，如果在途只有 1 个，整条流水线就退化成
    // “算一个叶子 → 等一次 GPU 往返 → 再算下一个”，GPU 批次小到几行、CPU 全程等。
    // 允许在途多个之后，一次前向能覆盖 lanes × in_flight 个叶子（再乘上每个
    // 展开节点自带的 5 个局面），CPU 搜索与 GPU 推理才真正重叠起来。
    uint32_t in_flight_per_lane = 8;
    float c_puct = 1.5f;              // PUCT 探索系数
    uint32_t temperature_moves = 30;  // 开局多少步按访问分布采样
    float temperature = 1.0f;         // 访问数开 1/T 次幂后采样；0 表示直接取最大
    // 搜索线程数（= K 个引擎，各持一份叶子缓存，共用一个写入侧与一个合并推理线程）。
    // C++ 侧**不再读它**——并行由 Python 侧驱动（见 ADR 0028）；保留在配置里是因为它会改变
    // 调度与批组成（ADR 0017），因此必须进协议参数与数据来源身份。
    uint32_t search_threads = 1;
    // 清单里记的"产出一批数据的并发对局数"：多线程搜索时它 = 各引擎 lanes 之和，
    // 而单个引擎的 lanes 只是它自己那份。0 = 就用 lanes（单引擎路径）。
    // 它进协议参数与数据来源身份，所以必须是**整批**的值而不是某个引擎的值。
    uint32_t lanes_total = 0;
    uint32_t cache_entries = 1 << 17;  // 推理缓存条目数（开放寻址表容量）
    float score_scale = kScoreScale;
    uint64_t seed = 0;
    DecisionMode decision_mode = kDecisionSearch;
    // 评测不需要训练数据：关掉之后引擎只推进对局并回报每局结果，不写任何分片。
    bool collect_samples = true;
    // >0 时按“跑满多少局”停止（评测用）；否则按样本目标停止（跑谱用）。
    uint32_t episodes_target = 0;
    // 控制文件路径（内容是 pause/stop）：跑谱要能被进程外叫停，评测不需要。
    std::string control_path;
    // 续跑：在既有数据批上接着编号写下去（参数逐项比对，不一致直接拒绝）。
    bool resume = false;
    // 加入一个**正在被别的写者写**的批次（多进程指向同一个 --out）：比对参数、从共享发号器
    // 领号、不读回末片。与 resume 互斥（resume 是"接上一次停下的位置顺序续写"）。
    bool join = false;
    // 产出侧才知道的协议参数（device、max_batch）：C++ 引擎不认识它们，但它们会改变
    // 逐局结果（批组成决定在途模拟的调度），所以必须作为来源记录写进清单，
    // 并在续跑时参与比对。Python 侧透传，不走猜测。
    std::vector<std::pair<std::string, std::string>> provenance;
};

// 一局结束的结果。评测只关心它，跑谱时它也用于进度显示。
struct EpisodeResult {
    uint32_t episode = 0;
    uint32_t steps = 0;
    uint32_t max_exponent = 0;
    uint64_t score = 0;
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
        return leaf_evaluations == 0
                   ? 0.0
                   : static_cast<double>(leaf_from_cache) / static_cast<double>(leaf_evaluations);
    }
};

// 收尾的结果。清单路径之外还回报这一批的终态：complete 表示跑满目标，
// paused/stopped 表示被叫停（此时可以带同样的参数 --continue 接着跑）。
struct FinishResult {
    std::string manifest_path;
    BatchState state = BatchState::kStopped;
    uint64_t samples = 0;
    uint32_t episodes = 0;
};

// 交互式会话的根局面报告：每个方向的访问数、边均值 Q、一次性估计（r + A）与先验，
// 外加根节点对 V(s) 的估计。字段与 `finish_step` 决策时读的是同一批数——所以
// "面板上看到的"与"引擎选招用到的"必然一致，不会出现两套口径。
struct RootReport {
    uint8_t legal_mask = 0;
    uint16_t visits[kNumActions] = {0, 0, 0, 0};
    // 含立即得分的边均值 Q；未访问的边退回一次性估计（与候选目标的写法一致）。
    float q_mean[kNumActions] = {0, 0, 0, 0};
    float one_ply[kNumActions] = {0, 0, 0, 0};
    float prior[kNumActions] = {0, 0, 0, 0};
    float root_value = 0.0f;
    uint32_t simulations = 0;  // 根实际完成的模拟数（时间上限下可能小于预算）
    int best_action = -1;      // 引擎此刻会走的那一步（与 commit_step 同一个选招）
};

// 一步提交的结果。**生成块由引擎自己生成并在这里回报**，所以"方块生成规则"仍然只有
// C++ 一处实现——会话（含人机对局的玩家模式）不必在 Python 侧再实现一遍。
struct StepResult {
    int action = -1;
    uint8_t board[kBoardCells] = {};
    int spawn_index = -1;  // 生成的块落在哪一格；终局或异常时为 -1
    uint8_t spawn_exponent = 0;
    uint64_t score_delta = 0;
    uint64_t total_score = 0;
    uint32_t step = 0;
    bool terminal = false;
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
//   * 并行放在“CPU 搜索 ∥ GPU 推理”这条流水线上：lane 之间顺序推进，但在途多条模拟
//     保证了每次前向都有几百行的批；实测 advance 只占管线时间约 5%（见 ADR 0003）。
class BatchedEngine {
 public:
    struct Advance {
        std::vector<uint64_t> request_ids;
        std::vector<uint8_t> request_kinds;
        std::vector<uint8_t> boards;          // N×16 行优先的指数棋盘
        std::vector<EpisodeResult> episodes;  // 自上次 advance 以来结束的局
        bool done = false;                    // 已完成目标，可以收尾
        uint64_t samples_written = 0;
        uint32_t episodes_written = 0;
    };

    BatchedEngine(const EngineConfig& config, std::string data_dir, std::string batch_id,
                  std::string model_id, uint64_t samples_target, uint64_t shard_samples);

    // 传入一个**共享的**写入侧：多线程搜索时 K 个引擎指向同一个 sink，于是它们产出的
    // 是同一批数据（一个批次号、一份清单、一套连续局号）。传 nullptr 时引擎自建一个。
    BatchedEngine(const EngineConfig& config, std::string data_dir, std::string batch_id,
                  std::string model_id, uint64_t samples_target, uint64_t shard_samples,
                  std::shared_ptr<SampleSink> sink);

    // 本引擎使用的写入侧。多线程驱动时用它把第一个引擎建好的 sink 分给其余引擎。
    const std::shared_ptr<SampleSink>& sink() const;
    ~BatchedEngine();

    BatchedEngine(const BatchedEngine&) = delete;
    BatchedEngine& operator=(const BatchedEngine&) = delete;

    // 推进所有 lane 直到需要神经网络，返回这一批请求。
    Advance advance(uint32_t max_requests);

    // 回填一批结果：values 为归一化后的价值，policy_logits 为 4 个方向的 logits
    // （按请求种类只取其一；长度均为 count）。未发出的 id 或重复回填直接抛错。
    void submit(const uint64_t* ids, const float* values, const float* policy_logits, size_t count);

    // 放弃所有在途请求：把它们重新排队（下次 advance 会再发一遍）。
    //
    // 契约：`advance` 交出去的一批，要么 `submit` 回填，要么在"确实丢了"的时候显式
    // `abandon`。**允许在未回填的情况下继续 advance**（管线重叠的前提：GPU 在算这一批时，
    // 搜索继续推进不依赖这些叶子的模拟），所以引擎不再自动补发——那会把正在计算的批次
    // 当成丢失的重复发一遍。异常路径（Ctrl-C 落在前向里、回调抛异常）由调用方调它。
    void abandon();

    // ---- 会话驱动：任意局面 + 逐步走（交互式分析与人机对局，见 ADR 0037）----------
    //
    // 与上面的**整局驱动互不影响**：并发跑局与评测的行为一字不变（既有调用点一行没改）。
    // 这一组只对 lane 0 生效，并要求 `lanes == 1` 且 `collect_samples == false`（违反即抛），
    // 因为会话是"一个盘面、一步一步走"的形态，而不是并发跑多局。
    //
    // 用法是三步循环：
    //     set_root(board)                    // 外部改变了盘面（摆棋、导入、undo）
    //     begin_analysis(budget) → 反复 advance/submit 直到 done
    //     report_root()                      // 看分析（不提交）
    //     commit_step(action)                // 走一步（action<0 = 由引擎从分析结果里选）
    //
    // 时间上限**不在这里**：它由调用方在 advance 循环里看表（到点就停手读 report_root），
    // 于是 C++ 不需要引入墙钟代码。时间上限下 report_root 给出的是"到目前为止"的结果，
    // 这由 `RootReport::simulations` 如实反映。

    // 把某个局面设为会话的当前根（替换开局派生）：摆棋、载入棋谱、undo 之后调用。
    // 会重置搜索树、步号与累计得分——它标记的是"盘面被外部改过"。
    void set_root(const Board& board);

    // 从**当前**根局面重新开始一次分析，预算为 simulations 次模拟；跑满后 advance 返回
    // done 且**不提交动作**（这正是"只分析"与"自动执行前先看一眼"共用的那一步）。
    void begin_analysis(uint32_t simulations);

    // 读当前根局面的统计。不推进搜索；只在 begin_analysis 之后（或在时间上限处中断后）有意义。
    RootReport report_root() const;

    // 走一步并提交：action >= 0 为指定方向（玩家模式），action < 0 表示由引擎按
    // `decision_mode` 从当前根（通常是刚分析完的那棵树）里选。
    StepResult commit_step(int action);

    EngineStats stats() const;

    // 请求在下一个合法停车点（局与局之间）停下并收尾。Python 侧的 KeyboardInterrupt
    // 走这条路：Ctrl-C 不该丢掉尾部已产出的数据。
    void request_stop();

    // 收尾结果里包含本次运行的终态：complete / paused / stopped。
    bool stop_requested() const;

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

    // 达到目标或被叫停后收尾：写完最后一个分片并落 manifest，返回 manifest 路径与终态。
    FinishResult finish();

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace r2048
