#pragma once

#include <cstdint>
#include <vector>

#include "core/rng.h"
#include "core/version.h"
#include "game/board.h"

namespace r2048 {

struct RolloutMctsConfig {
    uint32_t simulations = 64;
    // UCT 探索系数，作用于**节点内归一化**后的 Q。归一化是必要的：2048 的分数
    // 量级随局面演进从几十涨到几万，直接用原始 Q 会让探索项完全失去作用。
    float exploration = 1.0f;
    // rollout 的续弈策略：
    //   false（默认，历史行为）在合法方向里**均匀抽样**走到终局；
    //   true 用一步启发式（走完空位最多的方向）走到终局。
    //
    // 为什么需要这个开关：价值标签 `candidate_value = Q̄ − r` 就是在 rollout 里估出来的，
    // 而均匀随机的续弈会把局面质量差异洗掉——实测同根（同一局面的几个候选）标签的信号
    // std 只有约 36 分，与"这一手立刻合多少分"的 std（59~70 分）同量级，且它与真实回报的
    // 分层秩相关只有 0.05（ADR 0030）。把教师的搜索加深 4 倍（64→256 模拟）**没有**改善
    // 这些数字，所以限制在**续弈策略**上，不在搜索深度上。换一个"先把局面保住"的续弈策略
    // 是唯一非循环的改法（不引入任何网络）。
    bool greedy_rollout = false;
};

struct RolloutMctsResult {
    int action = -1;
    uint16_t visits[kNumActions] = {0, 0, 0, 0};
    float root_value = 0.0f;  // 根节点对 V(s) 的估计（从根局面起的未来得分）
    // 每条边的平均 Q（原始分数单位，含立即得分）。候选监督要用它算 A(u_a) = Q̄(a) − r(s,a)。
    // 未访问的边写 0：候选监督按 sqrt(访问数) 加权，权重 0 的候选自然被排除，
    // 不会把“没有估计”伪装成“估计为 0”。
    float q_mean[kNumActions] = {0, 0, 0, 0};
};

// UCT + 随机 rollout 的搜索（不含神经网络），用于冷启动第一批价值数据。
//
// 树结构：决策节点（轮到玩家）与随机块节点（afterstate，等一个随机块）交替。
//   * 决策节点用 UCT 选择动作，Q 在**节点内**归一化，并保证每条合法边至少被访问一次
//     （否则先验为 0 的边可能永远得不到探索）。
//   * 随机块节点没有“选择”可言，按真实规则采样一个生成结果，并把该结果作为 lazy
//     子节点缓存下来：同一个 (空格, 数值) 再次被采样到时复用已有子树。于是随机块
//     节点的价值是对 A(u) 的蒙特卡洛估计，且每次模拟最多只新增两个节点。
//   * 新展开的决策节点用均匀随机合法动作 rollout 到终局，返回值即从该局面起的未来得分。
//
// 这条备份路径让 Q(s,a) = r(s,a) + A(u) 成为树的自然结果，不需要额外修正项，
// 与 A(u)/Q/V 的价值语义严格一致。
//
// 所有随机流都由 (episode, step, 模拟序号) 派生，因此搜索结果与线程数、调度顺序无关，
// 这也是 verify 子命令能对整批数据做重放校验的前提。
class RolloutMcts {
public:
    RolloutMcts(const RolloutMctsConfig& config, uint64_t base_seed);

    RolloutMctsResult search(const Board& root, uint64_t episode, uint64_t step);

    size_t node_count() const { return nodes_.size(); }

private:
    static constexpr uint32_t kNoNode = 0xFFFFFFFFu;
    static constexpr int kMaxChanceChildren = 32;  // 16 个空格 × {2, 4}

    enum NodeKind : uint8_t { kDecision = 0, kChance = 1 };

    struct Node {
        Board board;                          // 决策节点：决策局面；随机块节点：afterstate
        uint8_t kind = kDecision;
        uint8_t legal_mask = 0;
        uint8_t parent_action = 0;
        uint8_t chance_children = 0;          // 随机块节点：已缓存的不同生成结果数
        uint32_t parent = kNoNode;
        uint32_t visits = 0;
        float value_sum = 0.0f;
        float edge_q_sum[kNumActions] = {0, 0, 0, 0};
        uint32_t edge_visits[kNumActions] = {0, 0, 0, 0};
        float edge_reward[kNumActions] = {0, 0, 0, 0};
        Board edge_afterstate[kNumActions];   // 决策节点：每条边的 afterstate（惰性填充）
        uint32_t edge_child[kNumActions] = {kNoNode, kNoNode, kNoNode, kNoNode};
        uint32_t chance_keys[kMaxChanceChildren] = {};   // (空格 << 1) | 是否为 4
        uint32_t chance_child[kMaxChanceChildren] = {};
    };

    uint32_t add_node(const Board& board, NodeKind kind, uint32_t parent, uint8_t parent_action);
    int select_action(const Node& node) const;
    float rollout(const Board& board, Rng& rng) const;
    // 一步启发式续弈：在合法方向里挑"走完之后空位最多"的，平局取立即得分高的。
    // 为什么是空位而不是立即得分：贪图立刻合并会把局面挤死（纯贪心实测只有约 2200 分），
    // 而空位数直接读作"还能活多久"——标签要的是"这个 afterstate 之后还能拿多少分"，
    // 续弈策略必须先把局面保住才谈得上分数。
    int greedy_action(const Board& board, uint8_t mask) const;

    RolloutMctsConfig config_;
    uint64_t base_seed_;
    std::vector<Node> nodes_;
};

}  // namespace r2048
