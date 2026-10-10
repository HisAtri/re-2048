#include <algorithm>
#include <cstdio>
#include <ctime>
#include <stdexcept>
#include <thread>

#include "command/command.h"
#include "core/args.h"
#include "core/json.h"
#include "core/logging.h"
#include "core/paths.h"
#include "program/evaluate.h"

namespace r2048 {

namespace {

void print_usage() {
    std::printf(
        "用法: r2048 evaluate --policy <random|mcts> --games <局数> [选项]\n\n"
        "评测：跑固定局数并汇总成绩。第 k 局在所有配置下都从同一开局开始，因此不同策略、\n"
        "不同模拟次数之间的对比是配对的。\n\n"
        "  --policy       random | mcts\n"
        "  --games        对局数\n"
        "  --seed         随机种子（不给则按当前时间生成并打印）\n"
        "  --simulations  mcts 每次决策的模拟次数（默认 64）\n"
        "  --workers      并行 worker 数（默认取硬件并发数）\n"
        "  --control      停止控制文件（局边界退出）\n"
        "  --jsonl        逐局结果写入的文件（默认 $R2048_OUTPUT_DIR/eval/<策略>-<时间>.jsonl）\n"
        "  --label        逐局记录里的标签（默认用策略名）\n"
        "  --progress-json 进度以 JSON 行写到 stderr（给上层编排读，不给人看）\n"
        "  --quiet        不打印计划行\n");
}

uint64_t default_seed() {
    return static_cast<uint64_t>(std::time(nullptr)) ^ 0x517cc1b727220a95ull;
}

std::string default_jsonl(const std::string& policy, uint32_t simulations) {
    std::time_t now = std::time(nullptr);
    std::tm local{};
    localtime_r(&now, &local);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &local);
    std::string name = policy;
    if (simulations > 0) name += "-s" + std::to_string(simulations);
    name += "-";
    name += stamp;
    name += ".jsonl";
    return default_eval_dir() + "/" + name;
}

}  // namespace

int run_evaluate(int argc, const char* const* argv) {
    Args args(argc, argv);
    if (args.flag("help")) {
        print_usage();
        return 0;
    }
    args.expect_only({"policy", "games", "seed", "simulations", "workers", "jsonl", "label", "quiet",
                      "temperature-moves", "progress-json", "control", "help"});

    EvaluateConfig config;
    config.policy = args.str("policy", "");
    if (config.policy.empty()) {
        print_usage();
        throw std::runtime_error("必须指定 --policy");
    }
    config.games = args.u64("games", 0);
    if (config.games == 0) {
        print_usage();
        throw std::runtime_error("必须指定 --games");
    }
    config.simulations = static_cast<uint32_t>(args.i32("simulations", 64));
    config.seed = args.has("seed") ? args.u64("seed", 0) : default_seed();
    config.workers = static_cast<uint32_t>(args.i32("workers", 0));
    // 评测默认不采样：确定性选招，避免把采样噪声混进策略对比。
    config.temperature_moves = static_cast<uint32_t>(args.i32("temperature-moves", 0));
    config.label = args.str("label", config.policy);
    config.jsonl_path = args.str("jsonl", default_jsonl(config.policy, config.simulations));
    config.progress_json = args.flag("progress-json");
    config.control_path = args.str("control", "");

    if (!args.flag("quiet")) {
        log_status(
            "计划：策略 %s%s，局数 %llu，种子 %llu，输出 %s", config.policy.c_str(),
            config.policy == "mcts" ? ("（模拟 " + std::to_string(config.simulations) + "）").c_str() : "",
            static_cast<unsigned long long>(config.games), static_cast<unsigned long long>(config.seed),
            config.jsonl_path.c_str());
    }

    const EvaluateResult result = run_evaluate(config);

    Json summary = Json::object();
    summary.set("label", Json::str(config.label));
    summary.set("policy", Json::str(config.policy));
    summary.set("simulations", Json::integer(config.simulations));
    summary.set("seed", Json::integer(static_cast<int64_t>(config.seed)));
    summary.set("games", Json::integer(static_cast<int64_t>(result.games.size())));
    summary.set("mean_score", Json::integer(static_cast<int64_t>(result.mean_score + 0.5)));
    summary.set("median_score", Json::integer(static_cast<int64_t>(result.median_score + 0.5)));
    summary.set("max_exponent", Json::integer(result.max_exponent));
    summary.set("seconds", Json::integer(static_cast<int64_t>(result.seconds + 0.5)));
    Json tiles = Json::object();
    for (const auto& entry : result.tile_counts) {
        tiles.set(std::to_string(1 << entry.first), Json::integer(entry.second));
    }
    summary.set("tiles", std::move(tiles));
    summary.set("jsonl", Json::str(config.jsonl_path));

    log_status("完成：均分 %.0f，中位 %.0f，最大块 2^%d，耗时 %.1f 秒（%.1f 局/秒）", result.mean_score,
               result.median_score, result.max_exponent, result.seconds,
               result.seconds > 0 ? static_cast<double>(result.games.size()) / result.seconds : 0.0);
    for (const auto& entry : result.tile_counts) {
        log_status("  达到 %d 的局数：%u / %zu", 1 << entry.first, entry.second, result.games.size());
    }
    std::printf("%s\n", summary.dump(0).c_str());
    return 0;
}

}  // namespace r2048
