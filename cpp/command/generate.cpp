#include <algorithm>
#include <cstdio>
#include <ctime>
#include <stdexcept>
#include <thread>

#include "command/command.h"
#include "core/args.h"
#include "core/logging.h"
#include "core/paths.h"
#include "program/generate.h"

namespace r2048 {

namespace {

void print_usage() {
    std::printf(
        "用法: r2048 generate --policy <random|mcts> --samples <条数> [选项]\n\n"
        "跑谱：自对弈生成训练数据。数据按分片落盘，并写入 manifest.json 作为来源记录。\n\n"
        "  --policy             random | mcts\n"
        "  --samples            目标样本条数（最后一个不完整的局会被截断到恰好这么多条）\n"
        "  --seed               随机种子；不给则按当前时间生成并打印（同一组参数与种子\n"
        "                       必须产出逐字节相同的数据，verify 会重放校验这一点）\n"
        "  --simulations        mcts 每次决策的模拟次数（默认 64）\n"
        "  --temperature-moves  开局多少步按根访问分布采样（默认 30；仅 mcts 有效）\n"
        "  --shard-samples      每个分片的样本数（默认 100000）\n"
        "  --workers            并行 worker 数（默认取硬件并发数）\n"
        "  --out                输出目录（默认 $R2048_OUTPUT_DIR/data/<批次号>）\n"
        "  --batch-id           批次号（默认由策略、模拟次数与时间生成）\n"
        "  --model-id           产出这批数据的模型标识（非 NN 策略留空）\n"
        "  --quiet              不打印计划行\n");
}

uint64_t default_seed() {
    return static_cast<uint64_t>(std::time(nullptr)) ^ 0x9e3779b97f4a7c15ull;
}

}  // namespace

int run_generate(int argc, const char* const* argv) {
    Args args(argc, argv);
    if (args.flag("help")) {
        print_usage();
        return 0;
    }
    args.expect_only({"policy", "samples", "seed", "simulations", "temperature-moves", "shard-samples",
                      "workers", "out", "batch-id", "model-id", "quiet", "help"});

    GenerateConfig config;
    config.policy = args.str("policy", "");
    if (config.policy.empty()) {
        print_usage();
        throw std::runtime_error("必须指定 --policy");
    }
    config.samples_target = args.u64("samples", 0);
    if (config.samples_target == 0) {
        print_usage();
        throw std::runtime_error("必须指定 --samples");
    }
    config.simulations = static_cast<uint32_t>(args.i32("simulations", 64));
    config.temperature_moves = static_cast<uint32_t>(args.i32("temperature-moves", 30));
    config.seed = args.has("seed") ? args.u64("seed", 0) : default_seed();
    config.shard_samples = args.u64("shard-samples", 100000);
    config.workers = static_cast<uint32_t>(args.i32("workers", 0));
    config.batch_id = args.str("batch-id", make_batch_id(config.policy, config.simulations));
    config.model_id = args.str("model-id", "");
    config.data_dir = args.str("out", default_data_dir(config.batch_id));

    // 原样记录命令行参数（输出目录除外，它由 manifest 的地点本身表达）：
    // 数据来源必须能从数据自身读出来，而不是靠翻命令行历史。
    for (const auto& item : args.items()) {
        if (item.first == "out" || item.first == "quiet" || item.first == "help") continue;
        config.params.push_back(item);
    }

    if (!args.flag("quiet")) {
        log_status("计划：策略 %s%s，样本 %llu，种子 %llu，worker %u，输出 %s", config.policy.c_str(),
                   config.policy == "mcts" ? ("（模拟 " + std::to_string(config.simulations) + "）").c_str() : "",
                   static_cast<unsigned long long>(config.samples_target),
                   static_cast<unsigned long long>(config.seed), config.workers == 0 ?
                       std::max(1u, std::thread::hardware_concurrency()) : config.workers,
                   config.data_dir.c_str());
    }

    const GenerateResult result = run_generate(config);
    const double per_second =
        result.seconds > 0 ? static_cast<double>(result.samples) / result.seconds : 0.0;
    log_status("完成：样本 %llu，局数 %u，耗时 %.2f 秒（%.0f 样本/秒）",
               static_cast<unsigned long long>(result.samples), result.episodes, result.seconds, per_second);
    // stdout 只放可被脚本消费的结果：manifest 路径。
    std::printf("%s\n", result.manifest_path.c_str());
    return 0;
}

}  // namespace r2048
