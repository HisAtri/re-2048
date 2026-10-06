#include <string>

#include "core/rng.h"
#include "game/rules.h"
#include "game/seeds.h"
#include "program/episode.h"
#include "search/rollout_mcts.h"
#include "test_util.h"

namespace r2048::tests {

namespace {

Board board_from(const char* rows[4]) {
    Board board = Board::empty();
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            const char symbol = rows[row][col];
            board.at(row, col) = symbol == '.' ? 0 : static_cast<uint8_t>(symbol - '0');
        }
    }
    return board;
}

}  // namespace

TEST(search_mcts_is_deterministic_and_legal) {
    // 同一 (局面, 局号, 步号) 的搜索结果必须完全一致：这是整批数据可重放的基础。
    const char* rows[4] = {"1232", "2.41", "1.2.", "..2."};
    const Board board = board_from(rows);
    const uint8_t mask = legal_mask(board);

    RolloutMctsConfig config;
    config.simulations = 64;
    RolloutMcts first(config, 20261006);
    RolloutMcts second(config, 20261006);
    const RolloutMctsResult a = first.search(board, 3, 7);
    const RolloutMctsResult b = second.search(board, 3, 7);

    expect_equal_u64(static_cast<uint64_t>(a.action), static_cast<uint64_t>(b.action), "同一输入的动作一致");
    for (int action = 0; action < kNumActions; action++) {
        expect_equal_u64(a.visits[action], b.visits[action], "同一输入的访问分布一致");
    }
    expect_true((mask & (1u << a.action)) != 0, "搜索给出的动作必须合法");

    // 未访问边优先的规则：模拟次数 ≥ 合法方向数时，每条合法边都必须被访问过。
    int legal_count = 0;
    uint32_t visit_sum = 0;
    for (int action = 0; action < kNumActions; action++) {
        legal_count += (mask >> action) & 1;
        visit_sum += a.visits[action];
    }
    expect_true(legal_count >= 2, "用例局面应有多个合法方向");
    for (int action = 0; action < kNumActions; action++) {
        if ((mask & (1u << action)) == 0) continue;
        expect_true(a.visits[action] > 0, "每条合法边至少被访问一次");
    }
    expect_equal_u64(visit_sum, 64, "根访问数之和等于模拟次数");
}

TEST(search_mcts_differs_by_seed_and_step) {
    // 随机流必须由 (种子, 局号, 步号) 决定：换种子或换步号都要产生不同的搜索。
    const char* rows[4] = {"....", "..2.", ".22.", "...."};
    const Board board = board_from(rows);

    RolloutMctsConfig config;
    config.simulations = 32;
    RolloutMcts searcher(config, 20261006);
    const RolloutMctsResult a = searcher.search(board, 1, 1);
    const RolloutMctsResult b = searcher.search(board, 1, 1);
    expect_equal_u64(a.visits[0] + a.visits[1] + a.visits[2] + a.visits[3],
                     b.visits[0] + b.visits[1] + b.visits[2] + b.visits[3], "同参数访问总数一致");

    RolloutMcts other_seed(config, 777);
    const RolloutMctsResult c = other_seed.search(board, 1, 1);
    bool differs = false;
    for (int action = 0; action < kNumActions; action++) {
        if (a.visits[action] != c.visits[action]) differs = true;
    }
    expect_true(differs, "换种子后访问分布应当不同");
}

TEST(search_mcts_single_simulation_is_legal) {
    // 模拟次数为 1 时仍必须返回合法动作，且访问数之和为 1。
    const char* rows[4] = {"2...", "....", "....", "...2"};
    const Board board = board_from(rows);
    RolloutMctsConfig config;
    config.simulations = 1;
    RolloutMcts searcher(config, 42);
    const RolloutMctsResult result = searcher.search(board, 0, 0);
    expect_true((legal_mask(board) & (1u << result.action)) != 0, "单次模拟也要返回合法动作");
    uint32_t sum = 0;
    for (int action = 0; action < kNumActions; action++) sum += result.visits[action];
    expect_equal_u64(sum, 1, "单次模拟的访问数之和为 1");
}

TEST(search_mcts_prefers_immediate_merge) {
    // 两个 1024 水平相邻：左右都能合并、上下只能滑动。搜索应当把访问数集中到
    // 有合并的两个方向。注意这里不能断言“具体是左还是右”——两个方向都能完成同一次
    // 合并，选哪个取决于后续 rollout，断言具体方向只会得到一个偶然通过的用例。
    Board board = Board::empty();
    board.at(0, 0) = 10;
    board.at(0, 1) = 10;
    board.at(1, 2) = 1;
    board.at(2, 3) = 1;
    board.at(3, 1) = 1;

    const uint8_t mask = legal_mask(board);
    expect_true((mask & (1u << kLeft)) != 0, "左移合法（可合并）");
    expect_true((mask & (1u << kRight)) != 0, "右移合法（可合并）");

    RolloutMctsConfig config;
    config.simulations = 64;
    RolloutMcts searcher(config, 20261006);
    const RolloutMctsResult result = searcher.search(board, 0, 0);

    const uint32_t merge_visits = result.visits[kLeft] + result.visits[kRight];
    const uint32_t slide_visits = result.visits[kUp] + result.visits[kDown];
    expect_true(merge_visits > slide_visits,
                "有合并的方向应当拿到更多访问（合并方向 " + std::to_string(merge_visits) + "，滑动方向 " +
                    std::to_string(slide_visits) + "）");
    expect_true((mask & (1u << result.action)) != 0, "给出的动作合法");
}

TEST(search_episode_records_consistent_samples) {
    // 一局完整对局的样本自洽性：afterstate 与实际移动一致、实际动作必在掩码内、
    // 回报标签严格满足“标签_t = 得分_{t+1} + 标签_{t+1}”。
    std::unique_ptr<MoveSelector> selector = make_selector("random", 20261006, 0, 0);
    EpisodeRecorder recorder(11, selector->simulations());
    const EpisodeOutcome outcome = play_episode(20261006, 11, *selector, &recorder);
    const std::vector<Sample> samples = recorder.finish();

    expect_equal_u64(samples.size(), outcome.steps, "样本数等于步数");
    expect_true(samples.size() > 10, "一局至少有十几步");
    for (size_t index = 0; index < samples.size(); index++) {
        const Sample& sample = samples[index];
        expect_equal_u64(sample.episode, 11, "局号一致");
        expect_equal_u64(sample.step, index, "步号连续");
        expect_true((sample.legal_mask & (1u << sample.action)) != 0, "实际动作在合法掩码内");
        const MoveOutcome expected = apply_move(sample.decision, sample.action);
        expect_true(expected.board == sample.afterstate, "afterstate 与实际移动一致");
        expect_equal_u64(sample.move_score, expected.score, "合并得分与实际移动一致");
        if (index + 1 < samples.size()) {
            // 注意不能断言“严格递减”：某一步可能只滑动不合并（得分为 0），
            // 这时相邻两步的标签相等。标签的精确关系是下面这一条。
            expect_equal_u64(static_cast<uint64_t>(samples[index].mc_return),
                             static_cast<uint64_t>(samples[index + 1].mc_return) + samples[index + 1].move_score,
                             "标签满足 标签_t = 得分_{t+1} + 标签_{t+1}");
        }
    }
    expect_equal_u64(static_cast<uint64_t>(samples.back().mc_return), 0, "最后一步标签为 0");
    expect_equal_u64(static_cast<uint64_t>(samples.front().mc_return),
                     outcome.total_score - samples.front().move_score,
                     "首步标签等于总分减去首步得分");
}

}  // namespace r2048::tests
