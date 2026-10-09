#include "dataio/sample.h"

#include <stdexcept>

#include "game/rules.h"

#include <array>
#include <cstring>

namespace r2048 {

namespace {

void write_u16(uint8_t* out, uint16_t value) {
    out[0] = static_cast<uint8_t>(value);
    out[1] = static_cast<uint8_t>(value >> 8);
}

void write_u32(uint8_t* out, uint32_t value) {
    for (int i = 0; i < 4; i++) out[i] = static_cast<uint8_t>(value >> (8 * i));
}

void write_u64(uint8_t* out, uint64_t value) {
    for (int i = 0; i < 8; i++) out[i] = static_cast<uint8_t>(value >> (8 * i));
}

uint16_t read_u16(const uint8_t* in) {
    return static_cast<uint16_t>(in[0]) | static_cast<uint16_t>(in[1] << 8);
}

uint32_t read_u32(const uint8_t* in) {
    uint32_t value = 0;
    for (int i = 3; i >= 0; i--) value = (value << 8) | in[i];
    return value;
}

uint64_t read_u64(const uint8_t* in) {
    uint64_t value = 0;
    for (int i = 7; i >= 0; i--) value = (value << 8) | in[i];
    return value;
}

}  // namespace

void encode_sample(const Sample& sample, uint8_t out[], uint8_t tree_nodes) {
    if (tree_nodes > kTreeNodesMax) {
        throw std::runtime_error("tree_nodes 超出编译期上界: " + std::to_string(tree_nodes));
    }
    std::memset(out, 0, sample_bytes(tree_nodes));
    std::memcpy(out + 0, sample.decision.tiles, 16);
    std::memcpy(out + 16, sample.afterstate.tiles, 16);
    out[32] = sample.action;
    out[33] = sample.legal_mask;
    write_u16(out + 34, sample.step);
    write_u32(out + 36, sample.episode);
    write_u16(out + 40, sample.simulations);
    write_u32(out + 42, sample.move_score);
    write_u64(out + 48, static_cast<uint64_t>(sample.mc_return));
    for (int action = 0; action < kNumActions; action++) write_u16(out + 64 + 2 * action, sample.visits[action]);

    // v2 候选块
    for (int action = 0; action < kNumActions; action++) {
        std::memcpy(out + 80 + 16 * action, sample.candidates.afterstate[action].tiles, 16);
        write_u32(out + 144 + 4 * action, static_cast<uint32_t>(sample.candidates.value[action]));
        write_u32(out + 160 + 4 * action, sample.candidates.reward[action]);
        write_u16(out + 176 + 2 * action, sample.candidates.visits[action]);
    }
    out[184] = sample.candidates.count;

    // v3 树内节点块：只写前 tree_count 个（K 声明了几个槽就写几个，其余保持 0）。
    const uint8_t nodes = std::min<uint8_t>(sample.tree_count, tree_nodes);
    // 实际写了几个节点**显式**记在基础块的保留字节 185 上（ADR 0007 留这块保留位就是为这个）。
    // 不靠"某个槽看起来是空的"去推断：真实决策节点必有合法方向，但"看起来空"与"没有这个节点"
    // 是两件事，推断出来的东西会在边界情况下静默错位。
    out[185] = nodes;
    for (uint8_t node = 0; node < nodes; node++) {
        const TreeNodeTarget& target = sample.tree[node];
        uint8_t* block = out + kSampleBaseBytes + static_cast<size_t>(node) * kTreeNodeBytes;
        block[0] = target.legal_mask;
        block[1] = target.count;
        block[2] = target.depth;
        for (int action = 0; action < kNumActions; action++) {
            std::memcpy(block + 44 + 16 * action, target.afterstate[action].tiles, 16);
            write_u32(block + 4 + 4 * action, static_cast<uint32_t>(target.value[action]));
            write_u32(block + 20 + 4 * action, target.reward[action]);
            write_u16(block + 36 + 2 * action, target.visits[action]);
        }
    }
}

Sample decode_sample(const uint8_t in[], uint8_t tree_nodes) {
    if (tree_nodes > kTreeNodesMax) {
        throw std::runtime_error("tree_nodes 超出编译期上界: " + std::to_string(tree_nodes));
    }
    Sample sample;
    std::memcpy(sample.decision.tiles, in + 0, 16);
    std::memcpy(sample.afterstate.tiles, in + 16, 16);
    sample.action = in[32];
    sample.legal_mask = in[33];
    sample.step = read_u16(in + 34);
    sample.episode = read_u32(in + 36);
    sample.simulations = read_u16(in + 40);
    sample.move_score = read_u32(in + 42);
    sample.mc_return = static_cast<int64_t>(read_u64(in + 48));
    for (int action = 0; action < kNumActions; action++) sample.visits[action] = read_u16(in + 64 + 2 * action);
    for (int action = 0; action < kNumActions; action++) {
        std::memcpy(sample.candidates.afterstate[action].tiles, in + 80 + 16 * action, 16);
        sample.candidates.value[action] = static_cast<int32_t>(read_u32(in + 144 + 4 * action));
        sample.candidates.reward[action] = read_u32(in + 160 + 4 * action);
        sample.candidates.visits[action] = read_u16(in + 176 + 2 * action);
    }
    sample.candidates.count = in[184];
    for (uint8_t node = 0; node < tree_nodes; node++) {
        TreeNodeTarget& target = sample.tree[node];
        const uint8_t* block = in + kSampleBaseBytes + static_cast<size_t>(node) * kTreeNodeBytes;
        target.legal_mask = block[0];
        target.count = block[1];
        target.depth = block[2];
        for (int action = 0; action < kNumActions; action++) {
            std::memcpy(target.afterstate[action].tiles, block + 44 + 16 * action, 16);
            target.value[action] = static_cast<int32_t>(read_u32(block + 4 + 4 * action));
            target.reward[action] = read_u32(block + 20 + 4 * action);
            target.visits[action] = read_u16(block + 36 + 2 * action);
        }
    }
    sample.tree_count = in[185];
    if (sample.tree_count > tree_nodes) {
        throw std::runtime_error("记录声明的树内节点数超过本批的 K: " +
                                 std::to_string(sample.tree_count));
    }
    return sample;
}

TreeNodeTarget build_tree_node(const Board& node_board, uint8_t legal_mask, uint8_t depth,
                               const int32_t q_mean_raw[kNumActions],
                               const uint16_t visits[kNumActions]) {
    TreeNodeTarget target;
    target.legal_mask = legal_mask;
    target.depth = depth;
    for (int action = 0; action < kNumActions; action++) {
        if ((legal_mask & (1u << action)) == 0) continue;
        const MoveOutcome outcome = apply_move(node_board, action);
        target.afterstate[action] = outcome.board;
        target.reward[action] = static_cast<uint32_t>(outcome.score);
        target.value[action] = q_mean_raw[action] - static_cast<int32_t>(outcome.score);
        target.visits[action] = visits[action];
        target.count++;
    }
    return target;
}

CandidateTargets build_candidate_targets(const Board& decision, uint8_t legal_mask,
                                         const int32_t q_mean_raw[kNumActions],
                                         const uint16_t visits[kNumActions]) {
    CandidateTargets targets;
    for (int action = 0; action < kNumActions; action++) {
        if ((legal_mask & (1u << action)) == 0) continue;
        const MoveOutcome outcome = apply_move(decision, action);
        targets.afterstate[action] = outcome.board;
        targets.reward[action] = static_cast<uint32_t>(outcome.score);
        targets.value[action] = q_mean_raw[action] - static_cast<int32_t>(outcome.score);
        targets.visits[action] = visits[action];
        targets.count++;
    }
    return targets;
}

EpisodeRecorder::EpisodeRecorder(uint32_t episode, uint16_t simulations)
    : episode_(episode), simulations_(simulations) {}

void EpisodeRecorder::add(const Board& decision, const Board& afterstate, int action, uint8_t legal_mask,
                          uint32_t move_score, const uint16_t visits[kNumActions],
                          const CandidateTargets* candidates, const TreeNodeTarget* tree,
                          uint8_t tree_count) {
    decision_.push_back(decision);
    afterstate_.push_back(afterstate);
    action_.push_back(static_cast<uint8_t>(action));
    legal_mask_.push_back(legal_mask);
    move_score_.push_back(move_score);
    std::array<uint16_t, kNumActions> copy{};
    for (int i = 0; i < kNumActions; i++) copy[i] = visits[i];
    visits_.push_back(copy);
    candidates_.push_back(candidates != nullptr ? *candidates : CandidateTargets{});
    std::array<TreeNodeTarget, kTreeNodesMax> nodes{};
    const uint8_t kept = std::min<uint8_t>(tree_count, kTreeNodesMax);
    for (uint8_t i = 0; i < kept && tree != nullptr; i++) nodes[i] = tree[i];
    tree_.push_back(nodes);
    tree_count_.push_back(kept);
}

std::vector<Sample> EpisodeRecorder::finish() {
    std::vector<Sample> samples;
    samples.reserve(decision_.size());

    // 逆序累加：最新一步的标签是 0（终局之后不再得分），
    // 往前每一步的标签是“之后所有合并得分之和”，不含本步的得分。
    int64_t future_score = 0;
    std::vector<int64_t> labels(decision_.size(), 0);
    for (size_t index = decision_.size(); index-- > 0;) {
        labels[index] = future_score;
        future_score += static_cast<int64_t>(move_score_[index]);
    }

    for (size_t index = 0; index < decision_.size(); index++) {
        Sample sample;
        sample.decision = decision_[index];
        sample.afterstate = afterstate_[index];
        sample.action = action_[index];
        sample.legal_mask = legal_mask_[index];
        sample.step = static_cast<uint16_t>(index);
        sample.episode = episode_;
        sample.simulations = simulations_;
        sample.move_score = move_score_[index];
        sample.mc_return = labels[index];
        for (int action = 0; action < kNumActions; action++) sample.visits[action] = visits_[index][action];
        sample.candidates = candidates_[index];
        sample.tree = tree_[index];
        sample.tree_count = tree_count_[index];
        samples.push_back(sample);
    }
    return samples;
}

}  // namespace r2048
