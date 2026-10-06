#include <cstdio>
#include <string>

#include "core/rng.h"
#include "game/rules.h"
#include "game/seeds.h"
#include "game/tables.h"
#include "rules_scalar.h"
#include "test_util.h"

namespace r2048::tests {

namespace {

const char* action_name(int action) {
    switch (action) {
        case kUp: return "UP";
        case kRight: return "RIGHT";
        case kDown: return "DOWN";
        default: return "LEFT";
    }
}

// 查表实现与慢速参考实现必须逐格、逐分一致。这条断言是规则层的唯一权威判据。
void check_matches_scalar(const Board& board, const std::string& context) {
    for (int action = 0; action < kNumActions; action++) {
        const MoveOutcome fast = apply_move(board, action);
        const MoveOutcome slow = scalar_apply_move(board, action);
        if (fast.valid != slow.valid || fast.score != slow.score || fast.board != slow.board) {
            fail("规则不一致 (" + context + ", 方向 " + action_name(action) + ")\n  棋盘 " +
                 board.to_string() + "\n  查表: " + fast.board.to_string() + " 分 " +
                 std::to_string(fast.score) + (fast.valid ? " 有效" : " 无效") + "\n  参考: " +
                 slow.board.to_string() + " 分 " + std::to_string(slow.score) +
                 (slow.valid ? " 有效" : " 无效"));
        }
    }
}

// 在指定行上放一条线，其余格清零。
Board board_with_row(int row, const uint8_t tiles[4]) {
    Board board = Board::empty();
    for (int i = 0; i < 4; i++) board.at(row, i) = tiles[i];
    return board;
}

Board board_with_col(int col, const uint8_t tiles[4]) {
    Board board = Board::empty();
    for (int i = 0; i < 4; i++) board.at(i, col) = tiles[i];
    return board;
}

}  // namespace

TEST(rules_known_cases) {
    // 有空格的行走线：_,2,4,8 左移变成 2,4,8,_，无合并。
    Board board = Board::empty();
    board.at(0, 1) = 1;
    board.at(0, 2) = 2;
    board.at(0, 3) = 3;
    const MoveOutcome shifted = apply_move(board, kLeft);
    expect_equal_u64(shifted.score, 0, "无合并的行得分为 0");
    expect_true(shifted.valid, "有空格的行左移是有效移动");
    expect_equal_u64(shifted.board.at(0, 0), 1, "2 被推到最左");
    expect_equal_u64(shifted.board.at(0, 2), 3, "8 被推到第 3 格");
    expect_equal_u64(shifted.board.at(0, 3), 0, "末格空出");

    // 填满且相邻块互不相同的行，左右都无法移动（既没有空格，也没有可合并的邻居）。
    Board packed = Board::empty();
    for (int i = 0; i < 4; i++) packed.at(0, i) = static_cast<uint8_t>(i + 1);
    expect_true(!apply_move(packed, kLeft).valid, "填满且无法合并的行左移无效");
    expect_true(!apply_move(packed, kRight).valid, "填满且无法合并的行右移同样无效");
    // 但只要出现一个可合并的邻居，左右就都变成有效移动。
    packed.at(0, 1) = 1;
    expect_true(apply_move(packed, kLeft).valid, "出现可合并邻居后左移有效");
    expect_true(apply_move(packed, kRight).valid, "出现可合并邻居后右移有效");

    board = Board::empty();
    board.at(0, 0) = 1;
    board.at(0, 1) = 1;
    board.at(0, 2) = 2;
    board.at(0, 3) = 2;
    // 2 2 4 4 → 4 8 _ _，得分 4 + 8 = 12。
    const MoveOutcome merged = apply_move(board, kLeft);
    expect_equal_u64(merged.score, 12, "两次合并的得分累加");
    expect_equal_u64(merged.board.at(0, 0), 2, "首格合并为 4");
    expect_equal_u64(merged.board.at(0, 1), 3, "次格合并为 8");

    board = Board::empty();
    board.at(0, 0) = 1;
    board.at(0, 1) = 2;
    board.at(0, 2) = 1;
    board.at(0, 3) = 2;
    // 2 4 2 4 左移不产生任何合并。
    expect_equal_u64(apply_move(board, kLeft).score, 0, "交替数值不合并");

    board = Board::empty();
    board.at(0, 0) = 2;
    board.at(0, 1) = 2;
    board.at(0, 2) = 2;
    board.at(0, 3) = 2;
    // 4 4 4 4 → 8 8 _ _：每次合并后的块不能再参与本回合的合并。
    const MoveOutcome one_pass = apply_move(board, kLeft);
    expect_equal_u64(one_pass.score, 16, "单次移动内每个块至多合并一次");
    expect_equal_u64(one_pass.board.at(0, 0), 3, "首格为 8");
    expect_equal_u64(one_pass.board.at(0, 1), 3, "次格为 8");

    // 右下角的单块：向右、向下都已贴边，无法移动；向左、向上仍可滑落。
    board = Board::empty();
    board.at(3, 3) = 1;
    const MoveOutcome stuck = apply_move(board, kRight);
    expect_true(!stuck.valid, "贴右下角的块不能再向右");
    expect_equal_u64(stuck.score, 0, "无效移动得分为 0");
    expect_true(stuck.board == board, "无效移动不改变棋盘");
    expect_true(!apply_move(board, kDown).valid, "贴右下角的块不能再向下");
    expect_true(apply_move(board, kLeft).valid, "贴右下角的块可以向左");
    expect_true(apply_move(board, kUp).valid, "贴右下角的块可以向上");
}

TEST(rules_terminal_detection) {
    // 满盘且相邻数各不相同 → 终局。
    Board board = Board::empty();
    const uint8_t pattern[16] = {1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1};
    for (int i = 0; i < 16; i++) board.tiles[i] = pattern[i];
    expect_true(is_terminal(board), "棋盘被填满且无法合并时为终局");
    expect_equal_u64(legal_mask(board), 0, "终局没有合法方向");

    board.at(0, 0) = 1;
    board.at(0, 1) = 1;
    expect_true(!is_terminal(board), "存在可合并的相邻块时不是终局");

    // 第一行四个相同的块：左右都能合并，向下能滑落，向上本已贴顶因此无效。
    // （“恰好只有一个方向合法”在结构上不可能：任一行可动时其垂直方向同样要考虑，
    //   因此这里断言一个可以手算验证的具体掩码，而不是不存在的单一方向。）
    Board row = Board::empty();
    for (int i = 0; i < 4; i++) row.at(0, i) = 1;
    const uint64_t expected =
        static_cast<uint64_t>((1u << kLeft) | (1u << kRight) | (1u << kDown));
    expect_equal_u64(legal_mask(row), expected, "首行四个 2 的合法方向集合");
    expect_true(!apply_move(row, kUp).valid, "贴顶的行不能再向上");
}

TEST(rules_exhaustive_lines) {
    // 单一行的全部组合（指数 0..4，5^4 = 625 种）分别放在每一行与每一列上：
    // 查表实现的两半拼接、跨半区合并、方向字节反转都在这个空间里被穷举到。
    uint8_t tiles[4];
    int checked = 0;
    for (int a = 0; a <= 4; a++) {
        for (int b = 0; b <= 4; b++) {
            for (int c = 0; c <= 4; c++) {
                for (int d = 0; d <= 4; d++) {
                    tiles[0] = static_cast<uint8_t>(a);
                    tiles[1] = static_cast<uint8_t>(b);
                    tiles[2] = static_cast<uint8_t>(c);
                    tiles[3] = static_cast<uint8_t>(d);
                    for (int index = 0; index < 4; index++) {
                        check_matches_scalar(board_with_row(index, tiles), "单行穷举");
                        check_matches_scalar(board_with_col(index, tiles), "单列穷举");
                        checked += 2;
                    }
                }
            }
        }
    }
    expect_true(checked == 625 * 8, "穷举覆盖数符合预期");
}

TEST(rules_random_boards) {
    // 随机棋盘覆盖多行多列互相影响、指数跨度大、近似满盘等情形。
    Rng rng(20261006);
    for (int iteration = 0; iteration < 200000; iteration++) {
        Board board = Board::empty();
        const int fill = static_cast<int>(rng.below(17));  // 0..16 个非空格
        for (int placed = 0; placed < fill; placed++) {
            const int index = static_cast<int>(rng.below(16));
            // 指数分布向小值倾斜：真实对局里大块只是少数。
            const int roll = static_cast<int>(rng.below(100));
            const uint8_t exponent = static_cast<uint8_t>(roll < 70 ? 1 + rng.below(6) : 1 + rng.below(17));
            board.tiles[index] = exponent;
        }
        check_matches_scalar(board, "随机棋盘");
        // legal_mask 与逐方向判定必须一致。
        uint8_t expected_mask = 0;
        for (int action = 0; action < kNumActions; action++) {
            if (apply_move(board, action).valid) expected_mask |= static_cast<uint8_t>(1u << action);
        }
        expect_equal_u64(legal_mask(board), expected_mask, "legal_mask 与逐方向判定一致");
    }
}

TEST(rules_spawn_rules) {
    // 生成块只落在空格上，数值只有 2（指数 1）与 4（指数 2），比例约 90:10。
    Rng rng(424242);
    Board board = Board::empty();
    board.tiles[5] = 7;
    board.tiles[11] = 3;
    int count_four = 0;
    int hits[16] = {0};
    const int samples = 200000;
    for (int i = 0; i < samples; i++) {
        Board copy = board;
        spawn_tile(copy, rng);
        int new_cells = 0;
        for (int cell = 0; cell < 16; cell++) {
            if (copy.tiles[cell] == board.tiles[cell]) continue;
            new_cells++;
            hits[cell]++;
            expect_true(copy.tiles[cell] == 1 || copy.tiles[cell] == 2, "生成块只能是 2 或 4");
            if (copy.tiles[cell] == 2) count_four++;
        }
        expect_equal_u64(new_cells, 1, "每次生成只产生一个新块");
    }
    const double four_ratio = static_cast<double>(count_four) / static_cast<double>(samples);
    expect_true(four_ratio > 0.085 && four_ratio < 0.115,
                "4 的比例应接近 10%，实测 " + std::to_string(four_ratio));
    // 14 个空格上应当近似均匀：不允许某些格子明显更容易被选中。
    const double expected_per_cell = static_cast<double>(samples) / 14.0;
    for (int cell = 0; cell < 16; cell++) {
        if (board.tiles[cell] != 0) {
            expect_equal_u64(hits[cell], 0, "生成块不能覆盖已有块");
            continue;
        }
        const double ratio = static_cast<double>(hits[cell]) / expected_per_cell;
        expect_true(ratio > 0.95 && ratio < 1.05, "空格被选中的频率应接近均匀");
    }
}

TEST(rules_seed_determinism) {
    // 相同派生种子必须产生完全相同的随机流；不同用途/局号/步号必须互相独立。
    const uint64_t a = derive_seed(1234, kSaltEnvironment, 7, 3, 0);
    const uint64_t b = derive_seed(1234, kSaltEnvironment, 7, 3, 0);
    expect_equal_u64(a, b, "同输入派生的种子相同");

    Rng rng_a(a);
    Rng rng_b(b);
    for (int i = 0; i < 32; i++) {
        expect_equal_u64(rng_a.next_u64(), rng_b.next_u64(), "相同种子产生相同随机流");
    }

    const uint64_t other_episode = derive_seed(1234, kSaltEnvironment, 8, 3, 0);
    expect_true(other_episode != a, "不同局号派生出不同种子");
    const uint64_t other_salt = derive_seed(1234, kSaltAction, 7, 3, 0);
    expect_true(other_salt != a, "不同用途派生出不同种子");

    // 静态相邻种子（0 与 1）经 splitmix64 展开后不能产生相关的流。
    Rng seed_zero(0);
    Rng seed_one(1);
    int matches = 0;
    for (int i = 0; i < 64; i++) {
        if (seed_zero.next_u64() == seed_one.next_u64()) matches++;
    }
    expect_equal_u64(matches, 0, "相邻种子的随机流不重合");

    // 开局复现：同一种子两次开局必须给出同一局面。
    Rng first(99);
    Rng second(99);
    const Board game_a = new_game(first);
    const Board game_b = new_game(second);
    expect_true(game_a == game_b, "同一随机种子的开局一致");
    int filled = 0;
    for (int i = 0; i < 16; i++) {
        if (game_a.tiles[i] != 0) {
            filled++;
            expect_true(game_a.tiles[i] == 1 || game_a.tiles[i] == 2, "开局只生成 2 或 4");
        }
    }
    expect_equal_u64(filled, 2, "开局生成两个块");
}

}  // namespace r2048::tests
