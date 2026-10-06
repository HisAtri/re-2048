#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "core/version.h"
#include "game/board.h"

namespace r2048 {

// 训练样本记录：**每条记录 = 一次真实决策的全部信息**，固定 80 字节。
//
// 设计取舍：旧版把 value 样本（64 字节）与 policy 样本（56 字节）分成两个文件流，
// 代价是每个批次都要校验“两条流按局号/步号对齐”，一旦错位很难发现。这里把一次
// 决策的所有字段放进同一条记录：决策局面（策略头）、afterstate（价值头）、实际
// 动作、根访问分布、回报标签，一次写入、一次校验。
//
// 偏移固定、小端、显式序列化（不依赖编译器对结构体的填充），因此格式只由这张表
// 决定。末尾的保留字节是给未来的 Q/V 头留的：加字段时不必改记录长度，
// 也就不必写迁移脚本（KataGo 因为自研权重格式付出了 9 个迁移脚本的代价）。
//
// 偏移  长度  字段
//   0    16   decision      决策局面 s（u8 指数）
//  16    16   afterstate    u = T(s, a)，移动后、生成随机块前
//  32     1   action        实际执行的方向 0..3
//  33     1   legal_mask    bit a = 1 表示方向 a 合法
//  34     2   step          局内步号（从 0 开始）
//  36     4   episode       局号
//  40     2   simulations   产出该样本时的搜索模拟次数（0 = 非搜索策略）
//  42     4   move_score    r(s, a)，本次移动的合并得分
//  46     2   保留（写 0）
//  48     8   mc_return     从 u 出发的实际回报，**不含** move_score
//  56     8   保留（写 0）
//  64     8   visits        四方向根访问数，u16 × 4（非搜索策略写 0）
//  72     8   保留（写 0）
constexpr size_t kSampleBytes = 80;
constexpr uint32_t kSampleSchemaVersion = 1;

struct Sample {
    Board decision;
    Board afterstate;
    uint8_t action = 0;
    uint8_t legal_mask = 0;
    uint16_t step = 0;
    uint32_t episode = 0;
    uint16_t simulations = 0;
    uint32_t move_score = 0;
    int64_t mc_return = 0;
    uint16_t visits[kNumActions] = {0, 0, 0, 0};
};

// 把样本序列化成 80 字节小端记录。写入与读取放在同一处，格式只有一份定义。
void encode_sample(const Sample& sample, uint8_t out[kSampleBytes]);
Sample decode_sample(const uint8_t in[kSampleBytes]);

// 一局的样本收集器：负责把“实际走过的动作”整理成训练样本，
// 并把回报标签按逆序累加（label_t = t 步之后还能拿到的分数，不含 t 步本身）。
//
// 这一步刻意与搜索解耦：无论是随机策略、UCT rollout 还是 NN 引导的搜索，
// 只要按顺序把 (决策局面, 动作, afterstate, 得分, visits) 交进来，标签口径就一致。
class EpisodeRecorder {
public:
    explicit EpisodeRecorder(uint32_t episode, uint16_t simulations);

    void add(const Board& decision, const Board& afterstate, int action, uint8_t legal_mask,
             uint32_t move_score, const uint16_t visits[kNumActions]);

    // 结算回报并产出样本；同一局的样本按步号升序返回。
    std::vector<Sample> finish();

    uint32_t episode() const { return episode_; }
    size_t steps() const { return decision_.size(); }

private:
    uint32_t episode_;
    uint16_t simulations_;
    std::vector<Board> decision_;
    std::vector<Board> afterstate_;
    std::vector<uint8_t> action_;
    std::vector<uint8_t> legal_mask_;
    std::vector<uint32_t> move_score_;
    std::vector<std::array<uint16_t, kNumActions>> visits_;
};

}  // namespace r2048
