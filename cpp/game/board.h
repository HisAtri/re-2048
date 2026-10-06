#pragma once

#include <cstdint>
#include <string>

#include "core/hash.h"
#include "core/version.h"

namespace r2048 {

// 4×4 棋盘：每格存“指数”（0 = 空，1 = 2，2 = 4，…）。
//
// 为什么是 u8 而不是把 4 个格打包进 16 位（nibble 表示）：nibble 只能表示到
// 2^15 = 32768，而本项目明确要研究 4096 及以上的大块（4×4 理论上界是 2^17），
// 表示层不能设人为上限。u8 覆盖到 2^255，规则实现因此不需要任何溢出特例。
// 16 字节也正好是一条 128 位比较指令的宽度，拷贝与比较和打包表示一样便宜。
struct Board {
    uint8_t tiles[kBoardCells];

    static Board empty();

    int empty_count() const;
    int max_exponent() const;
    uint8_t& at(int row, int col) { return tiles[row * kBoardSide + col]; }
    uint8_t at(int row, int col) const { return tiles[row * kBoardSide + col]; }

    bool operator==(const Board& other) const;
    bool operator!=(const Board& other) const { return !(*this == other); }

    uint64_t hash() const { return board_hash(tiles); }

    // 便于测试与日志的形状，例如 "2 4 0 0 | 0 0 0 0 | ..."。
    std::string to_string() const;

    // 行/列按字节打包成 u32，字节 0 是最左（列则是顶部）的格。
    // 查表移动统一在这个表示上做，方向差异只体现在打包顺序与字节反转上。
    uint32_t row_bits(int row) const;
    void set_row_bits(int row, uint32_t bits);
    uint32_t col_bits(int col) const;
    void set_col_bits(int col, uint32_t bits);
};

}  // namespace r2048
