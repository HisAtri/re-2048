#pragma once

#include <array>
#include <cstdint>

namespace r2048 {

// 半行（2 个格）左移的结果。
//
// 为什么按“半行”而不是整行查表：整行 4×u8 = 32 位，表大小 2^32 不可行；
// 半行 2×u8 = 16 位，表 65536 项、256KB，正好落在 L2。代价是拼接两半时要
// 处理跨半区的合并，规则见 move_row_left() 的注释。
struct HalfMove {
    uint16_t out;   // 左移后的两个格，左对齐（低字节是最左格）
    uint8_t len;    // 左移后占用的格数 0..2
    uint8_t flags;  // 见 HalfMoveFlags
};

enum HalfMoveFlags : uint8_t {
    kHalfMerged = 1u << 0,        // 半区内部发生过合并
    kHalfTailUnmerged = 1u << 1,  // 输出最后一个格是“未合并的单块”，可参与跨半区合并
};

extern const std::array<HalfMove, 65536> kHalfMoveTable;

struct RowMove {
    uint32_t row;    // 左移后的 4 个格
    uint64_t score;  // 本次左移的合并得分
};

// 对打包成 u32 的 4 个格做左移，等价于“压缩非零块 + 从左到右贪心合并”。
//
// 半区拼接的难点：整行贪心时，跨半区的那一对（低半区尾块、高半区首块）
// 优先级高于高半区内部的合并。例如 [_, 2, 2, 2] 应当先合并第 2、3 个格得到
// [4, 2]，而不是先把高半区合并成 [_, 4] 再拼接出 [2, 4]。
// 因此这里不用高半区“内部合并后”的首块做判断，而是取高半区**原始**的第一个
// 非零块，并在合并发生时改用“首块被吃掉”的结果。
RowMove move_row_left(uint32_t row_bits);

// 反转 4 个格的顺序，用于把“右移”化成“左移”。在小端字节序下就是 32 位字节反转。
inline uint32_t reverse_row_bits(uint32_t row_bits) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap32(row_bits);
#else
    return ((row_bits >> 24) & 0xff) | ((row_bits >> 8) & 0xff00) | ((row_bits << 8) & 0xff0000) |
           ((row_bits << 24) & 0xff000000);
#endif
}

}  // namespace r2048
