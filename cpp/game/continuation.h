#pragma once

#include <stdexcept>

#include "core/rng.h"
#include "game/board.h"
#include "game/rules.h"

namespace r2048 {

// 续弈策略：一个局面之后"继续怎么走"。它有两个使用者，因此必须只有一份实现：
//
//   1. rollout 引擎的模拟（`RolloutMcts::rollout`，纯 C++ 教师的叶值估计）；
//   2. 树内节点的**独立标注**（`BatchedEngine` 在抽样到的内部决策节点上跑 playout，
//      给价值头一个非循环的标签）。
//
// 为什么不能用第二份：ADR 0030 §6.1 实测它与标签信息量直接同源——把续弈策略从均匀换成
// 一步启发式（搜索深度完全不变），标签的同根 std 从 55.5 涨到 186.2、学生强度翻倍。两条
// 实现一旦漂开，"同一批数据里的标签口径"就不再是一个东西，而它看起来仍然正常。
enum class ContinuationPolicy : uint8_t { kUniform = 0, kGreedy = 1 };

// 一步启发式：走"走完空位最多"的方向（并列取立即得分更高者）。这是 ADR 0030 §6.1 里
// 被证实的那一条；它比均匀随机贵约 4.4 倍算力，但换来的标签区分度更高（按算力算更划算）。
inline int greedy_continuation(const Board& board, uint8_t mask) {
    int best = -1;
    int best_empty = -1;
    uint32_t best_score = 0;
    for (int action = 0; action < kNumActions; action++) {
        if ((mask & (1u << action)) == 0) continue;
        const MoveOutcome outcome = apply_move(board, action);
        if (!outcome.valid) continue;
        const int empties = outcome.board.empty_count();
        if (best < 0 || empties > best_empty || (empties == best_empty && outcome.score > best_score)) {
            best = action;
            best_empty = empties;
            best_score = outcome.score;
        }
    }
    if (best < 0) throw std::runtime_error("续弈里没有合法方向：入口应先用 legal_mask 过滤");
    return best;
}

// 按策略选一个方向。调用方负责保证 mask 非空。
inline int continuation_action(const Board& board, uint8_t mask, ContinuationPolicy policy, Rng& rng) {
    if (policy == ContinuationPolicy::kGreedy) return greedy_continuation(board, mask);
    // 均匀：先数掩码里的位置数，再取第 k 个。
    int picks = 0;
    for (int a = 0; a < kNumActions; a++) picks += (mask >> a) & 1;
    if (picks == 0) throw std::runtime_error("续弈里没有合法方向：入口应先用 legal_mask 过滤");
    int chosen = static_cast<int>(rng.below(static_cast<uint32_t>(picks)));
    for (int action = 0; action < kNumActions; action++) {
        if ((mask & (1u << action)) == 0) continue;
        if (chosen == 0) return action;
        chosen--;
    }
    throw std::runtime_error("续弈里没有合法方向：入口应先用 legal_mask 过滤");
}

// 从 board 一路走到终局，返回累计得分（不含 board 之前已经拿到的分）。
//
// 生成块的随机流由调用方给的 `rng` 决定：playout 内的随机块必须走**它自己的流**，
// 与真实环境的流分开，否则调整续弈策略会连带改变对局的随机序列（ADR 0006 的随机流约定）。
inline float playout(Board board, Rng& rng, ContinuationPolicy policy) {
    uint64_t total = 0;
    for (;;) {
        const uint8_t mask = legal_mask(board);
        if (mask == 0) break;
        const int action = continuation_action(board, mask, policy, rng);
        const MoveOutcome outcome = apply_move(board, action);
        total += outcome.score;
        board = outcome.board;
        spawn_tile(board, rng);
    }
    return static_cast<float>(total);
}

}  // namespace r2048
