#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "dataio/sample.h"
#include "game/board.h"
#include "search/rollout_mcts.h"

namespace r2048 {

struct EpisodeOutcome {
    uint64_t total_score = 0;
    int max_exponent = 0;
    uint32_t steps = 0;
};

// 决策器接口：给定决策局面与合法方向掩码，返回要执行的方向，并可附带根访问分布。
//
// 抽象出这一层是为了让随机策略、UCT-rollout、以及后续 NN 引导的搜索共用同一套
// 对局循环与样本收集逻辑，避免“每加一种策略就抄一遍对局循环”。
class MoveSelector {
public:
    virtual ~MoveSelector() = default;
    // candidates_out 非空时，决策器应当写入该局面的候选目标（候选级监督用）。
    // 非搜索策略（随机）没有候选目标，可以留空。
    virtual int select(const Board& board, uint8_t legal_mask, uint64_t episode, uint64_t step,
                       uint16_t visits_out[kNumActions],
                       CandidateTargets* candidates_out = nullptr) = 0;
    virtual uint16_t simulations() const { return 0; }
    virtual const char* name() const = 0;
};

// 均匀随机合法方向：产出纯随机行为策略的数据（速度极快，用于冷启动价值头）。
class RandomSelector : public MoveSelector {
public:
    explicit RandomSelector(uint64_t base_seed) : base_seed_(base_seed) {}
    int select(const Board& board, uint8_t legal_mask, uint64_t episode, uint64_t step,
               uint16_t visits_out[kNumActions], CandidateTargets* candidates_out = nullptr) override;
    const char* name() const override { return "random"; }

private:
    uint64_t base_seed_;
};

// UCT + 随机 rollout 的搜索策略。
class MctsSelector : public MoveSelector {
public:
    MctsSelector(uint64_t base_seed, const RolloutMctsConfig& config, uint32_t temperature_moves);

    int select(const Board& board, uint8_t legal_mask, uint64_t episode, uint64_t step,
               uint16_t visits_out[kNumActions], CandidateTargets* candidates_out = nullptr) override;
    uint16_t simulations() const override { return static_cast<uint16_t>(config_.simulations); }
    const char* name() const override { return "mcts"; }

private:
    uint64_t base_seed_;
    RolloutMctsConfig config_;
    uint32_t temperature_moves_;
    RolloutMcts search_;
};

// 开局棋盘：与步号无关的确定性开局（两次生成都用 step=0 的环境流）。
Board opening_board(uint64_t base_seed, uint64_t episode);

// 打一局完整对局。recorder 非空时按步收集训练样本。
EpisodeOutcome play_episode(uint64_t base_seed, uint32_t episode, MoveSelector& selector,
                            EpisodeRecorder* recorder);

// 按策略名构造决策器。未知策略名直接报错，不做静默回退。
// `greedy_rollout` 只对 mcts 有效：把 rollout 的续弈策略从均匀随机换成一步启发式
// （理由见 RolloutMctsConfig）。random 策略忽略它。
std::unique_ptr<MoveSelector> make_selector(const std::string& policy, uint64_t base_seed,
                                            uint32_t simulations, uint32_t temperature_moves,
                                            bool greedy_rollout = false);

}  // namespace r2048
