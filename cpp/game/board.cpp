#include "game/board.h"

#include <cstring>

namespace r2048 {

Board Board::empty() {
    Board board;
    std::memset(board.tiles, 0, sizeof(board.tiles));
    return board;
}

int Board::empty_count() const {
    int count = 0;
    for (int i = 0; i < kBoardCells; i++) count += (tiles[i] == 0) ? 1 : 0;
    return count;
}

int Board::max_exponent() const {
    int best = 0;
    for (int i = 0; i < kBoardCells; i++) best = tiles[i] > best ? tiles[i] : best;
    return best;
}

bool Board::operator==(const Board& other) const {
    return std::memcmp(tiles, other.tiles, sizeof(tiles)) == 0;
}

std::string Board::to_string() const {
    std::string out;
    for (int row = 0; row < kBoardSide; row++) {
        if (row > 0) out += " | ";
        for (int col = 0; col < kBoardSide; col++) {
            if (col > 0) out += ' ';
            const int exponent = at(row, col);
            out += exponent == 0 ? "0" : std::to_string(1u << (exponent < 30 ? exponent : 30));
        }
    }
    return out;
}

uint32_t Board::row_bits(int row) const {
    const uint8_t* p = tiles + row * kBoardSide;
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

void Board::set_row_bits(int row, uint32_t bits) {
    uint8_t* p = tiles + row * kBoardSide;
    p[0] = static_cast<uint8_t>(bits);
    p[1] = static_cast<uint8_t>(bits >> 8);
    p[2] = static_cast<uint8_t>(bits >> 16);
    p[3] = static_cast<uint8_t>(bits >> 24);
}

uint32_t Board::col_bits(int col) const {
    return static_cast<uint32_t>(tiles[col]) | (static_cast<uint32_t>(tiles[col + 4]) << 8) |
           (static_cast<uint32_t>(tiles[col + 8]) << 16) | (static_cast<uint32_t>(tiles[col + 12]) << 24);
}

void Board::set_col_bits(int col, uint32_t bits) {
    tiles[col] = static_cast<uint8_t>(bits);
    tiles[col + 4] = static_cast<uint8_t>(bits >> 8);
    tiles[col + 8] = static_cast<uint8_t>(bits >> 16);
    tiles[col + 12] = static_cast<uint8_t>(bits >> 24);
}

}  // namespace r2048
