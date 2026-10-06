#pragma once

#include <cstdint>

#include "game/board.h"
#include "game/rules.h"

namespace r2048::tests {

// 规则语义的“慢速参考实现”，只用于测试，不参与生产路径。
//
// 它按最直白的方式实现规则：把一条线（行或列）上的非零块压缩，再从左到右
// 贪心合并。查表实现的全部复杂度都是为了在不改变语义的前提下把它变快，
// 因此必须有一个不共享任何代码的版本作为判据 —— 否则表里的错误会被
// “用同一张表算出来的期望值”掩盖过去。
inline uint64_t scalar_move_line(const uint8_t in[4], uint8_t out[4]) {
    uint8_t packed[4];
    int count = 0;
    for (int i = 0; i < 4; i++) {
        if (in[i] != 0) packed[count++] = in[i];
    }
    uint8_t merged[4] = {0, 0, 0, 0};
    uint64_t score = 0;
    int written = 0;
    int cursor = 0;
    while (cursor < count) {
        if (cursor + 1 < count && packed[cursor] == packed[cursor + 1]) {
            const uint8_t exponent = static_cast<uint8_t>(packed[cursor] + 1);
            merged[written++] = exponent;
            score += 1ull << exponent;
            cursor += 2;
        } else {
            merged[written++] = packed[cursor];
            cursor++;
        }
    }
    for (int i = 0; i < 4; i++) out[i] = merged[i];
    return score;
}

// 一条线的取格顺序与 Python 参考实现完全一致：
//   LEFT  = 每行从左到右        RIGHT = 每行从右到左
//   UP    = 每列从上到下        DOWN  = 每列从下到上
inline void scalar_line_indices(int action, int index, int cells[4]) {
    switch (action) {
        case kLeft:
            for (int i = 0; i < 4; i++) cells[i] = index * 4 + i;
            break;
        case kRight:
            for (int i = 0; i < 4; i++) cells[i] = index * 4 + (3 - i);
            break;
        case kUp:
            for (int i = 0; i < 4; i++) cells[i] = i * 4 + index;
            break;
        default:  // kDown
            for (int i = 0; i < 4; i++) cells[i] = (3 - i) * 4 + index;
            break;
    }
}

inline MoveOutcome scalar_apply_move(const Board& board, int action) {
    Board out = board;
    uint64_t score = 0;
    for (int line = 0; line < 4; line++) {
        int cells[4];
        scalar_line_indices(action, line, cells);
        uint8_t in[4];
        for (int i = 0; i < 4; i++) in[i] = board.tiles[cells[i]];
        uint8_t moved[4];
        score += scalar_move_line(in, moved);
        for (int i = 0; i < 4; i++) out.tiles[cells[i]] = moved[i];
    }
    return {out, score, out != board};
}

}  // namespace r2048::tests
