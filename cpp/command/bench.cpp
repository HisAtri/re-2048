#include <cstdio>
#include <vector>

#include "command/command.h"
#include "core/args.h"
#include "core/json.h"
#include "core/logging.h"
#include "core/rng.h"
#include "game/rules.h"
#include "game/seeds.h"
#include "search/rollout_mcts.h"

namespace r2048 {

namespace {

void print_usage() {
    std::printf(
        "用法: r2048 bench [--positions N] [--episodes N] [--simulations N] [--seed S]\n\n"
        "微基准：把“移动生成”“随机对局”“UCT-rollout 搜索”三条路径的吞吐分别测出来。\n"
        "在改棋盘表示、查表或 rollout 之前先跑一次，改动之后再跑一次——ADR 里的性能结论\n"
        "都应当能用这里的数字复现。\n");
}

// 用一批随机局面测移动生成：每个局面走完四个方向，统计每秒能处理多少次“移动+生成块”。
double bench_move_generation(uint64_t seed, int positions) {
    Rng rng(seed);
    std::vector<Board> boards;
    boards.reserve(static_cast<size_t>(positions));
    for (int i = 0; i < positions; i++) {
        Board board = Board::empty();
        const int fill = 4 + static_cast<int>(rng.below(12));
        for (int placed = 0; placed < fill; placed++) {
            board.tiles[rng.below(16)] = static_cast<uint8_t>(1 + rng.below(10));
        }
        boards.push_back(board);
    }

    const double start = monotonic_seconds();
    const int repeats = 200;
    uint64_t checksum = 0;
    for (int repeat = 0; repeat < repeats; repeat++) {
        for (const Board& board : boards) {
            for (int action = 0; action < kNumActions; action++) {
                const MoveOutcome outcome = apply_move(board, action);
                checksum += outcome.score + outcome.board.tiles[0];
            }
        }
    }
    const double seconds = monotonic_seconds() - start;
    const double moves = static_cast<double>(repeats) * static_cast<double>(positions) * kNumActions;
    log_status("移动生成：%.1f 万次/秒（checksum %llu 仅为防止被优化掉）", moves / seconds / 1e4,
               static_cast<unsigned long long>(checksum));
    return moves / seconds;
}

// 随机合法动作自对弈一局（不含搜索），测“对局推进 + 生成块”的吞吐。
double bench_random_episodes(uint64_t seed, int episodes) {
    const double start = monotonic_seconds();
    uint64_t total_steps = 0;
    for (int episode = 0; episode < episodes; episode++) {
        Rng env_rng(derive_seed(seed, kSaltEnvironment, static_cast<uint64_t>(episode), 0, 0));
        Board board = new_game(env_rng);
        for (uint64_t step = 0;; step++) {
            const uint8_t mask = legal_mask(board);
            if (mask == 0) break;
            Rng action_rng(derive_seed(seed, kSaltAction, static_cast<uint64_t>(episode), step, 0));
            int picks = 0;
            for (int action = 0; action < kNumActions; action++) picks += (mask >> action) & 1;
            int chosen = static_cast<int>(action_rng.below(static_cast<uint32_t>(picks)));
            int action = 0;
            for (; action < kNumActions; action++) {
                if ((mask & (1u << action)) == 0) continue;
                if (chosen == 0) break;
                chosen--;
            }
            board = apply_move(board, action).board;
            Rng spawn_rng(derive_seed(seed, kSaltEnvironment, static_cast<uint64_t>(episode), step + 1, 0));
            spawn_tile(board, spawn_rng);
            total_steps++;
        }
    }
    const double seconds = monotonic_seconds() - start;
    const double samples = static_cast<double>(total_steps);
    log_status("随机对局：%.1f 万步/秒（%d 局共 %llu 步）", samples / seconds / 1e4, episodes,
               static_cast<unsigned long long>(total_steps));
    return samples / seconds;
}

// UCT + rollout 搜索：统计每秒完成的模拟次数（每次模拟含一条到终局的 rollout）。
double bench_rollout_search(uint64_t seed, uint32_t simulations, int positions) {
    RolloutMctsConfig config;
    config.simulations = simulations;
    RolloutMcts search(config, seed);

    Rng rng(seed ^ 0x5bf03635ull);
    std::vector<Board> boards;
    for (int i = 0; i < positions; i++) {
        Rng env(derive_seed(seed, kSaltEnvironment, static_cast<uint64_t>(i), 0, 0));
        Board board = Board::empty();
        const int fill = 4 + static_cast<int>(rng.below(10));
        for (int placed = 0; placed < fill; placed++) {
            board.tiles[rng.below(16)] = static_cast<uint8_t>(1 + rng.below(8));
        }
        if (legal_mask(board) == 0) continue;
        boards.push_back(board);
    }

    const double start = monotonic_seconds();
    uint64_t simulations_done = 0;
    for (const Board& board : boards) {
        search.search(board, 0, simulations_done);
        simulations_done += simulations;
    }
    const double seconds = monotonic_seconds() - start;
    log_status("UCT-rollout 搜索：%.0f 次模拟/秒（%zu 个局面 × %u 次模拟）",
               static_cast<double>(simulations_done) / seconds, boards.size(), simulations);
    return static_cast<double>(simulations_done) / seconds;
}

}  // namespace

int run_bench(int argc, const char* const* argv) {
    Args args(argc, argv);
    if (args.flag("help")) {
        print_usage();
        return 0;
    }
    args.expect_only({"positions", "episodes", "simulations", "seed", "help"});

    const int positions = args.i32("positions", 2000);
    const int episodes = args.i32("episodes", 400);
    const uint32_t simulations = static_cast<uint32_t>(args.i32("simulations", 64));
    const uint64_t seed = args.u64("seed", 20261006);

    const double moves_per_second = bench_move_generation(seed, positions);
    const double steps_per_second = bench_random_episodes(seed, episodes);
    const double sims_per_second = bench_rollout_search(seed, simulations, positions / 10 + 1);

    Json summary = Json::object();
    summary.set("positions", Json::integer(positions));
    summary.set("episodes", Json::integer(episodes));
    summary.set("simulations", Json::integer(simulations));
    summary.set("seed", Json::integer(static_cast<int64_t>(seed)));
    summary.set("moves_per_second", Json::integer(static_cast<int64_t>(moves_per_second)));
    summary.set("steps_per_second", Json::integer(static_cast<int64_t>(steps_per_second)));
    summary.set("simulations_per_second", Json::integer(static_cast<int64_t>(sims_per_second)));
    std::printf("%s\n", summary.dump(0).c_str());
    return 0;
}

}  // namespace r2048
