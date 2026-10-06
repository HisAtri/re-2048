#pragma once

#include <cstdint>

#include "core/rng.h"
#include "game/board.h"

namespace r2048 {

// 记 s 为“随机块已生成、轮到玩家移动”的决策局面，a 为方向，
// u = T(s, a) 为移动后、生成随机块前的 afterstate，r(s, a) 为本次移动的合并得分。
// 本文件只负责这三件事，价值语义（A(u)/Q/V）属于搜索层。
struct MoveOutcome {
    Board board;     // afterstate；无效移动时等于输入棋盘
    uint64_t score;  // 本次移动的合并得分 r(s, a)
    bool valid;      // 是否有效移动（无效移动不生成新块，也不改变局面）
};

MoveOutcome apply_move(const Board& board, int action);

// 4 位掩码，bit a = 1 表示方向 a 合法。
uint8_t legal_mask(const Board& board);
bool is_terminal(const Board& board);

// 在随机空格生成新块：位置在空格上均匀，指数 1（即 2）占 90%、指数 2（即 4）占 10%。
// 调用方必须先确认存在空格：终局棋盘生成块属于上层逻辑错误，不做静默兜底。
void spawn_tile(Board& board, Rng& rng);

// 开局：空棋盘上连续生成两个块。
Board new_game(Rng& rng);

}  // namespace r2048
