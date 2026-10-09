#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "core/version.h"
#include "game/board.h"

namespace r2048 {

// 训练样本记录：**每条记录 = 一次真实决策的全部信息**。
//
// 设计取舍：旧版把 value 样本与 policy 样本分成两个文件流，还要为“候选监督”再开第三条
// sidecar 流，代价是每条记录必须跨流校验局号/步号/动作集合是否对齐。这里把一次决策的
// 所有信息放进同一条记录：决策局面、afterstate、实际动作、根访问分布、回报标签，
// 以及**每个合法动作的候选目标**（候选级监督，见下）。一次写入、一次校验。
//
// 为什么候选监督必须进主记录：搜索在每个决策节点要查询**全部合法动作**的 afterstate，
// 而主轨迹只记录实际走的那个动作——价值头因此在自己被查询的地方没有监督。实测的后果是
// 价值头在贪心决策上退化到随机以下（评分 858 vs 随机 1000），并且自举迭代会把它的标度
// 训坏（同一个空开局的预测从 1020 变成 8885）。候选块就是为消除这个失配而存在的。
//
// 偏移固定、小端、整数编码（不依赖编译器对结构体的填充，也不用浮点以免端序歧义）。
// 末尾保留字节是给未来字段留的：加字段时不必改记录长度，也就不必写迁移脚本。
//
// v2 布局（v1 的前 80 字节及其偏移完全不变，因此版本闸门是关键——旧的 80 字节批次会被
// 明确拒绝，而不会被当成 v2 误读）：
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
// --- v2：候选块（按方向下标对齐，槽位 a 恒对应方向 a）---
//  80    64   candidate_afterstate[4]  每个合法动作的 afterstate（16 × 4）
// 144    16   candidate_value[4]       int32：Q̄(a) − r(s,a)，**原始分数单位**（四舍五入到 1 分）
// 160    16   candidate_reward[4]      uint32：r(s,a)
// 176     8   candidate_visits[4]      u16 × 4：该候选边的根访问数
// 184     1   candidate_count          0 = 无候选监督；否则等于 legal_mask 的位数
// 185     7   保留（写 0）
//
// 标签语义（与 A(u) 严格同构）：引擎每条边的 Q 累加的是 `r(s,a) + A(u_a)`，
// 因此 `Q̄(a) − r(s,a)` 就是该候选 afterstate 的价值估计 A(u_a) 本身。
// 量纲是原始分数（Python 侧再除以 score_scale），与主轨迹的 mc_return 同一量纲。
//
// --- v3：树内节点块（**可选**，节点数 K 由清单的 `tree_nodes` 声明，K=0 时记录就是 192 字节）---
// 紧跟在候选块之后，每个节点 108 字节，节点顺序按"访问数从大到小、并列取节点下标小者"
// （确定性，重放校验依赖它）：
//   偏移  长度  字段
//     0     1   legal_mask    该节点决策局面的合法掩码
//     1     1   count         有效候选槽数（= legal_mask 的位数）
//     2     1   depth         该节点到根的步数（根 = 0）
//     3     1   保留（写 0，4 字节对齐用）
//     4    16   value[4]      int32：Q̄(a) − r(s,a)，与候选块同定义、同量纲
//    20    16   reward[4]     uint32：r(s,a)
//    36     8   visits[4]     u16 × 4：该节点上这条边的访问数
//    44    64   afterstate[4] 每个合法动作的 afterstate（16 × 4）
//  节点块不需要存节点自己的 decision 盘面：价值头的输入是 afterstate，而 afterstate 就在块里。
//
// 为什么要有这一块（ADR 0041）：价值头在搜索里被**直接使用**的位置是叶节点与未被访问的边
// （那里的 Q 就是 `r + A_θ`，没有模拟平均来纠错），而根候选块只覆盖"轨迹根"——搜索在根上有
// 几百次模拟，自己就把排序算出来了，价值头在根上的排序改进会被它覆盖（ADR 0035 实测的正是
// 这一点：监督轨迹根能显著提升贪心臂，但转移给搜索的收益很小）。树内节点访问数少、平均弱，
// 是"价值头真正说了算"的地方。
constexpr size_t kSampleBaseBytes = 192;
// 一个树内节点块的字节数（见上面的偏移表）。
constexpr size_t kTreeNodeBytes = 108;
// 单条记录里最多放几个树内节点。K 由清单声明、可以更小；这是编译期上界（定长数组）。
constexpr uint8_t kTreeNodesMax = 4;
constexpr size_t sample_bytes(uint8_t tree_nodes) {
    return kSampleBaseBytes + static_cast<size_t>(tree_nodes) * kTreeNodeBytes;
}
constexpr uint32_t kSampleSchemaVersion = 3;

// 一个决策局面的候选目标集合。搜索负责给出每条边的平均 Q 与访问数，
// afterstate 与立即得分由样本构造方用规则算（只有一个规则实现，避免两条路径分叉）。
struct CandidateTargets {
    uint8_t count = 0;
    // 按方向下标存放；仅合法方向有效。
    std::array<Board, kNumActions> afterstate{};
    std::array<int32_t, kNumActions> value{0, 0, 0, 0};    // (Q̄ − r)，原始分数单位
    std::array<uint32_t, kNumActions> reward{0, 0, 0, 0};  // r(s, a)
    std::array<uint16_t, kNumActions> visits{0, 0, 0, 0};  // 该候选边的根访问数
};

// 搜索树内部一个**决策节点**的快照。搜索负责给出每条边的平均 Q 与访问数、以及节点的
// 合法掩码与深度；afterstate 与立即得分由规则算（与候选块同一处实现，避免两条路径分叉）。
struct TreeNodeTarget {
    uint8_t legal_mask = 0;
    uint8_t count = 0;
    uint8_t depth = 0;
    std::array<Board, kNumActions> afterstate{};
    std::array<int32_t, kNumActions> value{0, 0, 0, 0};
    std::array<uint32_t, kNumActions> reward{0, 0, 0, 0};
    std::array<uint16_t, kNumActions> visits{0, 0, 0, 0};
};

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
    std::array<uint16_t, kNumActions> visits{0, 0, 0, 0};
    CandidateTargets candidates;
    // 树内节点块：只写前 tree_count 个槽，其余留空（K 由清单声明）。
    std::array<TreeNodeTarget, kTreeNodesMax> tree{};
    uint8_t tree_count = 0;
};

// 把样本序列化成定长小端记录。写入与读取放在同一处，格式只有一份定义。
// `tree_nodes` 是本批声明的 K（0..kTreeNodesMax），它决定记录的**字节数**——
// 因此它必须来自清单而不是被猜：读侧按清单声明的 K 解码，K 不符时记录长度就错了（ADR 0007）。
void encode_sample(const Sample& sample, uint8_t out[], uint8_t tree_nodes);
Sample decode_sample(const uint8_t in[], uint8_t tree_nodes);

// 由一个树内决策节点构造节点块：afterstate 与立即得分用规则补齐，与候选块同一处实现。
TreeNodeTarget build_tree_node(const Board& node_board, uint8_t legal_mask, uint8_t depth,
                               const int32_t q_mean_raw[kNumActions],
                               const uint16_t visits[kNumActions]);

// 由决策局面与（每条边的平均 Q、访问数）构造候选目标。
// afterstate 与立即得分在这里用规则补齐，因此调用方只需提供搜索结果。
CandidateTargets build_candidate_targets(const Board& decision, uint8_t legal_mask,
                                         const int32_t q_mean_raw[kNumActions],
                                         const uint16_t visits[kNumActions]);

// 一局的样本收集器：负责把“实际走过的动作”整理成训练样本，
// 并把回报标签按逆序累加（label_t = t 步之后还能拿到的分数，不含 t 步本身）。
class EpisodeRecorder {
public:
    explicit EpisodeRecorder(uint32_t episode, uint16_t simulations);

    void add(const Board& decision, const Board& afterstate, int action, uint8_t legal_mask,
             uint32_t move_score, const uint16_t visits[kNumActions],
             const CandidateTargets* candidates = nullptr, const TreeNodeTarget* tree = nullptr,
             uint8_t tree_count = 0);

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
    std::vector<CandidateTargets> candidates_;
    // 这一步搜索的树内节点快照（最多 kTreeNodesMax 个）；空表示这一步没采集。
    std::vector<std::array<TreeNodeTarget, kTreeNodesMax>> tree_;
    std::vector<uint8_t> tree_count_;
};

}  // namespace r2048
