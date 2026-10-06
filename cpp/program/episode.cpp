#include "program/episode.h"

#include <cmath>

#include <stdexcept>

#include "core/rng.h"
#include "game/rules.h"
#include "game/seeds.h"

namespace r2048 {

namespace {

int pick_uniform_legal(uint8_t mask, Rng& rng) {
    int picks = 0;
    for (int action = 0; action < kNumActions; action++) picks += (mask >> action) & 1;
    int chosen = static_cast<int>(rng.below(static_cast<uint32_t>(picks)));
    for (int action = 0; action < kNumActions; action++) {
        if ((mask & (1u << action)) == 0) continue;
        if (chosen == 0) return action;
        chosen--;
    }
    throw std::runtime_error("合法方向掩码与抽样结果不一致");
}

}  // namespace

int RandomSelector::select(const Board& board, uint8_t legal_mask, uint64_t episode, uint64_t step,
                           uint16_t visits_out[kNumActions], CandidateTargets* candidates_out) {
    (void)board;
    (void)candidates_out;  // 非搜索策略没有候选估计
    for (int action = 0; action < kNumActions; action++) visits_out[action] = 0;
    Rng rng(derive_seed(base_seed_, kSaltAction, episode, step, 0));
    return pick_uniform_legal(legal_mask, rng);
}

MctsSelector::MctsSelector(uint64_t base_seed, const RolloutMctsConfig& config, uint32_t temperature_moves)
    : base_seed_(base_seed),
      config_(config),
      temperature_moves_(temperature_moves),
      search_(config, base_seed) {}

int MctsSelector::select(const Board& board, uint8_t legal_mask, uint64_t episode, uint64_t step,
                         uint16_t visits_out[kNumActions], CandidateTargets* candidates_out) {
    const RolloutMctsResult result = search_.search(board, episode, step);
    for (int action = 0; action < kNumActions; action++) visits_out[action] = result.visits[action];
    if (candidates_out != nullptr) {
        int32_t q_mean[kNumActions] = {0, 0, 0, 0};
        for (int action = 0; action < kNumActions; action++) {
            q_mean[action] = static_cast<int32_t>(std::lround(result.q_mean[action]));
        }
        *candidates_out = build_candidate_targets(board, legal_mask, q_mean, result.visits);
    }

    // 开局若干步按根访问分布采样而不是直接取最大值：自对弈数据需要多样性，
    // 全程 argmax 会让同一批数据高度重复。第 temperature_moves 步之后转为确定性地
    // 取访问数最大的方向（访问数相同时取编号更小的方向，保证可复现）。
    if (step < temperature_moves_) {
        uint32_t total = 0;
        for (int action = 0; action < kNumActions; action++) total += result.visits[action];
        if (total > 0) {
            Rng rng(derive_seed(base_seed_, kSaltAction, episode, step, 0));
            uint32_t pick = rng.below(total);
            for (int action = 0; action < kNumActions; action++) {
                if (result.visits[action] == 0) continue;
                if (pick < result.visits[action]) return action;
                pick -= result.visits[action];
            }
        }
    }

    int best_action = -1;
    uint16_t best_visits = 0;
    for (int action = 0; action < kNumActions; action++) {
        if ((legal_mask & (1u << action)) == 0) continue;
        if (result.visits[action] > best_visits) {
            best_visits = result.visits[action];
            best_action = action;
        }
    }
    if (best_action < 0) throw std::runtime_error("搜索没有给出任何合法方向");
    return best_action;
}

Board opening_board(uint64_t base_seed, uint64_t episode) {
    // 开场两个块用 step=0 的环境流；第 t 步之后的生成块用 step=t+1 的流，
    // 因此“开局”和“任意一步的随机块”都不会共享随机序列。
    Rng rng(derive_seed(base_seed, kSaltEnvironment, episode, 0, 0));
    return new_game(rng);
}

EpisodeOutcome play_episode(uint64_t base_seed, uint32_t episode, MoveSelector& selector,
                            EpisodeRecorder* recorder) {
    Board board = opening_board(base_seed, episode);
    EpisodeOutcome outcome;
    outcome.max_exponent = board.max_exponent();

    uint64_t step = 0;
    for (;;) {
        const uint8_t mask = legal_mask(board);
        if (mask == 0) break;

        uint16_t visits[kNumActions] = {0, 0, 0, 0};
        CandidateTargets candidates;
        const int action = selector.select(board, mask, episode, step, visits, &candidates);
        const MoveOutcome move = apply_move(board, action);
        if (!move.valid) {
            throw std::runtime_error("决策器返回了非法方向 " + std::to_string(action));
        }
        if (recorder != nullptr) {
            recorder->add(board, move.board, action, mask, static_cast<uint32_t>(move.score), visits,
                          &candidates);
        }
        outcome.total_score += move.score;
        board = move.board;

        Rng rng(derive_seed(base_seed, kSaltEnvironment, episode, step + 1, 0));
        spawn_tile(board, rng);

        const int exponent = board.max_exponent();
        outcome.max_exponent = exponent > outcome.max_exponent ? exponent : outcome.max_exponent;
        step++;
    }
    outcome.steps = static_cast<uint32_t>(step);
    return outcome;
}

std::unique_ptr<MoveSelector> make_selector(const std::string& policy, uint64_t base_seed,
                                            uint32_t simulations, uint32_t temperature_moves) {
    if (policy == "random") return std::make_unique<RandomSelector>(base_seed);
    if (policy == "mcts") {
        RolloutMctsConfig config;
        config.simulations = simulations;
        return std::make_unique<MctsSelector>(base_seed, config, temperature_moves);
    }
    throw std::runtime_error("未知策略: " + policy + "（可选 random / mcts）");
}

}  // namespace r2048
