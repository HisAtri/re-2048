#include "search/rollout_mcts.h"

#include <cfloat>
#include <cmath>
#include <stdexcept>

#include "game/continuation.h"
#include "game/rules.h"
#include "game/seeds.h"

namespace r2048 {

RolloutMcts::RolloutMcts(const RolloutMctsConfig& config, uint64_t base_seed)
    : config_(config), base_seed_(base_seed) {
    if (config_.simulations == 0) throw std::runtime_error("模拟次数必须大于 0");
    if (config_.simulations > 0xFFFF) throw std::runtime_error("模拟次数上限为 65535（根访问数按 u16 落盘）");
    nodes_.reserve(static_cast<size_t>(config_.simulations) * 2 + 4);
}

uint32_t RolloutMcts::add_node(const Board& board, NodeKind kind, uint32_t parent, uint8_t parent_action) {
    Node node;
    node.board = board;
    node.kind = static_cast<uint8_t>(kind);
    node.parent = parent;
    node.parent_action = parent_action;
    node.legal_mask = kind == kDecision ? legal_mask(board) : 0;
    nodes_.push_back(node);
    return static_cast<uint32_t>(nodes_.size() - 1);
}

int RolloutMcts::select_action(const Node& node) const {
    // 未访问过的合法边一律优先：保证每条边至少被评估一次。
    for (int action = 0; action < kNumActions; action++) {
        if ((node.legal_mask & (1u << action)) != 0 && node.edge_visits[action] == 0) return action;
    }

    float min_q = FLT_MAX;
    float max_q = -FLT_MAX;
    for (int action = 0; action < kNumActions; action++) {
        if ((node.legal_mask & (1u << action)) == 0) continue;
        const float q = node.edge_q_sum[action] / static_cast<float>(node.edge_visits[action]);
        min_q = q < min_q ? q : min_q;
        max_q = q > max_q ? q : max_q;
    }
    const float range = max_q - min_q;
    const float log_visits = std::log(static_cast<float>(node.visits > 1 ? node.visits : 1));

    int best_action = -1;
    float best_score = -FLT_MAX;
    for (int action = 0; action < kNumActions; action++) {
        if ((node.legal_mask & (1u << action)) == 0) continue;
        const float q = node.edge_q_sum[action] / static_cast<float>(node.edge_visits[action]);
        const float normalized = range > 1e-6f ? (q - min_q) / range : 0.0f;
        const float exploration =
            config_.exploration * std::sqrt(log_visits / static_cast<float>(node.edge_visits[action]));
        const float score = normalized + exploration;
        if (score > best_score) {
            best_score = score;
            best_action = action;
        }
    }
    if (best_action < 0) throw std::runtime_error("决策节点没有合法方向：搜索入口应先用 legal_mask 过滤");
    return best_action;
}

int RolloutMcts::greedy_action(const Board& board, uint8_t mask) const {
    // 续弈策略只有一份实现（game/continuation.h）：rollout 引擎与树内节点的独立标注共用它。
    // 分成两份的话它们会漂，而续弈策略正是标签信息量的决定因素（ADR 0030 §6.1、ADR 0043）。
    return greedy_continuation(board, mask);
}

float RolloutMcts::rollout(const Board& board, Rng& rng) const {
    const ContinuationPolicy policy =
        config_.greedy_rollout ? ContinuationPolicy::kGreedy : ContinuationPolicy::kUniform;
    return playout(board, rng, policy);
}

RolloutMctsResult RolloutMcts::search(const Board& root, uint64_t episode, uint64_t step) {
    nodes_.clear();
    const uint8_t root_mask = legal_mask(root);
    if (root_mask == 0) throw std::runtime_error("终局局面不能进入搜索");

    const uint32_t root_index = add_node(root, kDecision, kNoNode, 0);

    for (uint32_t simulation = 0; simulation < config_.simulations; simulation++) {
        // 环境流与 rollout 流分开派生：前者只服务树内的随机块采样，后者只服务 rollout。
        Rng spawn_rng(derive_seed(base_seed_, kSaltEnvironment, episode, step, 1 + simulation));
        Rng rollout_rng(derive_seed(base_seed_, kSaltRollout, episode, step, simulation));

        uint32_t node_index = root_index;
        float leaf_value = 0.0f;

        for (;;) {
            if (nodes_[node_index].kind == kChance) {
                Node& chance = nodes_[node_index];
                const int empties = chance.board.empty_count();
                if (empties == 0) {
                    // 有效移动必定至少空出一格，走到这里说明 afterstate 状态被破坏了。
                    throw std::runtime_error("随机块节点没有空格：afterstate 状态不合法");
                }
                int cell_pick = static_cast<int>(spawn_rng.below(static_cast<uint32_t>(empties)));
                int cell = 0;
                for (; cell < kBoardCells; cell++) {
                    if (chance.board.tiles[cell] != 0) continue;
                    if (cell_pick == 0) break;
                    cell_pick--;
                }
                const bool is_four = spawn_rng.one_in_ten();
                const uint32_t key = static_cast<uint32_t>(cell) * 2u + (is_four ? 1u : 0u);

                uint32_t child = kNoNode;
                for (int i = 0; i < chance.chance_children; i++) {
                    if (chance.chance_keys[i] == key) {
                        child = chance.chance_child[i];
                        break;
                    }
                }
                if (child != kNoNode) {
                    node_index = child;
                    continue;
                }

                Board spawned = chance.board;
                spawned.tiles[cell] = is_four ? 2 : 1;
                child = add_node(spawned, kDecision, node_index, 0);
                // add_node 之后引用可能失效，必须重新取。
                Node& updated = nodes_[node_index];
                updated.chance_keys[updated.chance_children] = key;
                updated.chance_child[updated.chance_children] = child;
                updated.chance_children++;
                leaf_value = rollout(nodes_[child].board, rollout_rng);
                node_index = child;
                break;  // 新展开的节点就是本次模拟的叶子
            }

            Node& decision = nodes_[node_index];
            if (decision.legal_mask == 0) {
                leaf_value = 0.0f;  // 终局：此后不再得分
                break;
            }
            const int action = select_action(decision);
            uint32_t child = decision.edge_child[action];
            if (child == kNoNode) {
                // 首次走这条边：惰性计算 afterstate 与立即得分，然后创建随机块节点。
                const MoveOutcome outcome = apply_move(decision.board, action);
                child = add_node(outcome.board, kChance, node_index, static_cast<uint8_t>(action));
                Node& updated = nodes_[node_index];
                updated.edge_child[action] = child;
                updated.edge_afterstate[action] = outcome.board;
                updated.edge_reward[action] = static_cast<float>(outcome.score);
                updated.edge_visits[action]++;
                node_index = child;
                continue;
            }
            decision.edge_visits[action]++;
            node_index = child;
        }

        // 回传：从叶子沿路径向上累加。穿过“随机块节点 → 决策父节点”时把该边的立即
        // 合并得分计入返回值，于是父节点的 Q 自然等于 r(s,a) + A(u)。
        uint32_t node = node_index;
        float value = leaf_value;
        for (;;) {
            const Node& current = nodes_[node];
            const uint32_t parent_index = current.parent;
            const uint8_t parent_action = current.parent_action;
            Node& mutable_current = nodes_[node];
            mutable_current.visits++;
            mutable_current.value_sum += value;
            if (parent_index == kNoNode) break;
            Node& parent = nodes_[parent_index];
            if (parent.kind == kDecision) {
                parent.edge_q_sum[parent_action] += parent.edge_reward[parent_action] + value;
                value += parent.edge_reward[parent_action];
            }
            node = parent_index;
        }
    }

    RolloutMctsResult result;
    const Node& final_root = nodes_[root_index];
    float best_q = -FLT_MAX;
    for (int action = 0; action < kNumActions; action++) {
        if ((root_mask & (1u << action)) == 0) continue;
        const uint32_t visits = final_root.edge_visits[action];
        result.visits[action] = static_cast<uint16_t>(visits > 0xFFFF ? 0xFFFF : visits);
        if (visits > 0) {
            const float q = final_root.edge_q_sum[action] / static_cast<float>(visits);
            if (q > best_q) {
                best_q = q;
                result.action = action;
            }
        }
    }
    if (result.action < 0) throw std::runtime_error("搜索没有产生任何合法动作");
    for (int action = 0; action < kNumActions; action++) {
        if ((root_mask & (1u << action)) == 0) continue;
        const uint32_t visits = final_root.edge_visits[action];
        if (visits > 0) {
            result.q_mean[action] = final_root.edge_q_sum[action] / static_cast<float>(visits);
        }
    }
    result.root_value =
        final_root.visits > 0 ? final_root.value_sum / static_cast<float>(final_root.visits) : 0.0f;
    return result;
}

}  // namespace r2048
