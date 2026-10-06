#include "game/rules.h"

#include <stdexcept>

#include "game/tables.h"

namespace r2048 {

namespace {

// 四个方向共用一条实现：行优先/列优先只影响打包方式，反向只影响是否先做一次
// 字节反转。这样“方向”不会变成四份可能互相分叉的规则代码。
template <bool kByRow, bool kReversed>
void apply_lines(const Board& src, Board& dst, uint64_t& score) {
    for (int i = 0; i < kBoardSide; i++) {
        uint32_t bits = kByRow ? src.row_bits(i) : src.col_bits(i);
        if (kReversed) bits = reverse_row_bits(bits);
        const RowMove moved = move_row_left(bits);
        score += moved.score;
        const uint32_t out = kReversed ? reverse_row_bits(moved.row) : moved.row;
        if (kByRow) {
            dst.set_row_bits(i, out);
        } else {
            dst.set_col_bits(i, out);
        }
    }
}

}  // namespace

MoveOutcome apply_move(const Board& board, int action) {
    Board out = board;
    uint64_t score = 0;
    switch (action) {
        case kLeft:
            apply_lines<true, false>(board, out, score);
            break;
        case kRight:
            apply_lines<true, true>(board, out, score);
            break;
        case kUp:
            apply_lines<false, false>(board, out, score);
            break;
        case kDown:
            apply_lines<false, true>(board, out, score);
            break;
        default:
            throw std::runtime_error("非法方向: " + std::to_string(action));
    }
    return {out, score, out != board};
}

uint8_t legal_mask(const Board& board) {
    uint8_t mask = 0;
    for (int action = 0; action < kNumActions; action++) {
        if (apply_move(board, action).valid) mask |= static_cast<uint8_t>(1u << action);
    }
    return mask;
}

bool is_terminal(const Board& board) { return legal_mask(board) == 0; }

void spawn_tile(Board& board, Rng& rng) {
    const int empties = board.empty_count();
    if (empties == 0) throw std::runtime_error("棋盘已满：spawn_tile 只能用于还有空格的局面");
    int pick = static_cast<int>(rng.below(static_cast<uint32_t>(empties)));
    for (int i = 0; i < kBoardCells; i++) {
        if (board.tiles[i] != 0) continue;
        if (pick == 0) {
            board.tiles[i] = rng.one_in_ten() ? 2 : 1;
            return;
        }
        pick--;
    }
}

Board new_game(Rng& rng) {
    Board board = Board::empty();
    spawn_tile(board, rng);
    spawn_tile(board, rng);
    return board;
}

}  // namespace r2048
