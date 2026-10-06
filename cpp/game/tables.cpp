#include "game/tables.h"

namespace r2048 {

namespace {

constexpr HalfMove compute_half_move(uint16_t half) {
    const uint8_t left = static_cast<uint8_t>(half);
    const uint8_t right = static_cast<uint8_t>(half >> 8);
    if (left == 0 && right == 0) return {0, 0, 0};
    if (left == 0) return {right, 1, kHalfTailUnmerged};   // [_, b] → [b, _]
    if (right == 0) return {left, 1, kHalfTailUnmerged};   // [a, _] → [a, _]
    if (left == right) return {static_cast<uint16_t>(left + 1), 1, kHalfMerged};  // [a, a] → [a+1, _]
    // [a, b]（a≠b）两个单块都原样保留，尾块还能与半区外的右邻合并。
    return {half, 2, kHalfTailUnmerged};
}

constexpr std::array<HalfMove, 65536> build_half_move_table() {
    std::array<HalfMove, 65536> table{};
    for (uint32_t i = 0; i < 65536; i++) table[i] = compute_half_move(static_cast<uint16_t>(i));
    return table;
}

}  // namespace

// constinit 保证这张表是静态常量初始化（.rodata）：进程启动不花时间，
// 也不存在“第一次调用时初始化”的隐藏同步点。
constinit const std::array<HalfMove, 65536> kHalfMoveTable = build_half_move_table();

RowMove move_row_left(uint32_t row_bits) {
    const HalfMove lo = kHalfMoveTable[row_bits & 0xffffu];
    const HalfMove hi = kHalfMoveTable[(row_bits >> 16) & 0xffffu];

    uint64_t score = 0;
    // 低半区的内部合并与跨半区合并互不冲突（它发生在更左边），可以无条件计入。
    if ((lo.flags & kHalfMerged) != 0) score += 1ull << (lo.out & 0xffu);

    uint8_t out[4] = {0, 0, 0, 0};
    int pos = 0;
    for (int i = 0; i < lo.len; i++) out[pos++] = static_cast<uint8_t>(lo.out >> (8 * i));

    const uint8_t hi_left = static_cast<uint8_t>(row_bits >> 16);
    const uint8_t hi_right = static_cast<uint8_t>(row_bits >> 24);
    // 高半区原始的第一个非零块：压缩之后它就是低半区尾块的右邻。
    const uint8_t hi_first = hi_left != 0 ? hi_left : hi_right;

    if (lo.len > 0 && (lo.flags & kHalfTailUnmerged) != 0 && hi_first != 0 &&
        out[lo.len - 1] == hi_first) {
        // 跨半区合并，优先于高半区内部的合并。
        out[lo.len - 1] = static_cast<uint8_t>(out[lo.len - 1] + 1);
        score += 1ull << out[lo.len - 1];
        // 首块被吃掉后，高半区最多剩一个块，直接接在后面。
        const uint8_t survivor = hi_left != 0 ? hi_right : 0;
        if (survivor != 0) out[pos++] = survivor;
    } else {
        // 高半区的内部合并只有在没有发生跨半区合并时才成立，得分也只能记一次。
        if ((hi.flags & kHalfMerged) != 0) score += 1ull << (hi.out & 0xffu);
        for (int i = 0; i < hi.len; i++) out[pos++] = static_cast<uint8_t>(hi.out >> (8 * i));
    }

    const uint32_t packed = static_cast<uint32_t>(out[0]) | (static_cast<uint32_t>(out[1]) << 8) |
                            (static_cast<uint32_t>(out[2]) << 16) | (static_cast<uint32_t>(out[3]) << 24);
    return {packed, score};
}

}  // namespace r2048
