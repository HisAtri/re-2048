#include "dataio/sample.h"

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

void encode_sample(const Sample& sample, uint8_t out[kSampleBytes]) {
    std::memset(out, 0, kSampleBytes);
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
}

Sample decode_sample(const uint8_t in[kSampleBytes]) {
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
    return sample;
}

EpisodeRecorder::EpisodeRecorder(uint32_t episode, uint16_t simulations)
    : episode_(episode), simulations_(simulations) {}

void EpisodeRecorder::add(const Board& decision, const Board& afterstate, int action, uint8_t legal_mask,
                          uint32_t move_score, const uint16_t visits[kNumActions]) {
    decision_.push_back(decision);
    afterstate_.push_back(afterstate);
    action_.push_back(static_cast<uint8_t>(action));
    legal_mask_.push_back(legal_mask);
    move_score_.push_back(move_score);
    std::array<uint16_t, kNumActions> copy{};
    for (int i = 0; i < kNumActions; i++) copy[i] = visits[i];
    visits_.push_back(copy);
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
        samples.push_back(sample);
    }
    return samples;
}

}  // namespace r2048
