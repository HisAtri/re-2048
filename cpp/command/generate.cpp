#include <algorithm>
#include <cstdio>
#include <ctime>
#include <stdexcept>
#include <thread>

#include "command/command.h"
#include "core/args.h"
#include "core/control.h"
#include "core/logging.h"
#include "core/paths.h"
#include "program/generate.h"

namespace r2048 {

namespace {

void print_usage() {
    std::printf(
        "用法: r2048 generate --policy <random|mcts> [--samples <条数>] [选项]\n\n"
        "跑谱：自对弈生成训练数据。数据按分片落盘，并写入 manifest.json 作为来源记录。\n"
        "清单是**增量提交日志**：每落一个分片就原子重写一次，被强杀只会留下未被列出的\n"
        "孤儿分片，不会污染这批数据。\n\n"
        "  --policy             random | mcts\n"
        "  --samples            目标样本条数；0 或不给表示**不限量**，跑到被叫停为止\n"
        "                       （最后一个不完整的局会被截断到恰好这么多条）\n"
        "  --seed               随机种子；不给则按当前时间生成并打印（同一组参数与种子\n"
        "                       必须产出逐字节相同的数据，verify 会重放校验这一点）\n"
        "  --simulations        mcts 每次决策的模拟次数（默认 64）\n"
        "  --rollout-policy     uniform | greedy：rollout 的续弈策略（默认 uniform）。\n"
        "                       greedy = 一步启发式（走完空位最多的方向）走到终局；\n"
        "                       它决定候选价值标签的信息量（ADR 0030），属协议参数\n"
        "  --temperature-moves  开局多少步按根访问分布采样（默认 30；仅 mcts 有效）\n"
        "  --shard-samples      每个分片的样本数（默认 100000）\n"
        "  --workers            并行 worker 数（默认取硬件并发数）\n"
        "  --out                输出目录（默认 $R2048_OUTPUT_DIR/data/<批次号>）\n"
        "  --batch-id           批次号（默认由策略、模拟次数与时间生成）\n"
        "  --model-id           产出这批数据的模型标识（非 NN 策略留空）\n"
        "  --continue           续跑：在既有批次上接着编号写下去（参数必须逐项一致）\n"
        "  --control            控制文件路径（默认 <数据批目录>/control，内容是 pause/stop）\n"
        "  --progress-json      进度以 JSON 行输出到 stderr（供脚本与界面解析）\n"
        "  --quiet              不打印计划行\n\n"
        "停车与续跑：Ctrl-C（SIGINT/SIGTERM）等价于写 stop 到控制文件——进行中的局走完、\n"
        "已产出的数据提交、清单记下这次是 paused 还是 stopped，然后正常退出。\n"
        "续跑产出的分片与一次跑完的分片逐字节相同（第 N 局只由种子与局号决定）。\n");
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
    args.expect_only({"policy", "samples", "seed", "simulations", "temperature-moves", "tree-nodes",
                      "rollout-policy",
                      "shard-samples", "workers", "out", "batch-id", "model-id", "continue", "control",
                      "progress-json", "quiet", "help"});

    GenerateConfig config;
    config.policy = args.str("policy", "");
    if (config.policy.empty()) {
        print_usage();
        throw std::runtime_error("必须指定 --policy");
    }
    config.samples_target = args.u64("samples", 0);  // 0 = 不限量
    config.simulations = static_cast<uint32_t>(args.i32("simulations", 64));
    config.temperature_moves = static_cast<uint32_t>(args.i32("temperature-moves", 30));
    // 树内节点块（ADR 0041）：它是**格式参数**，不是逐局协议参数——采集不改变任何一局的走法，
    // 但它决定记录的字节数，所以进清单、并且续跑时必须一致（见 compare_resume_meta）。
    config.tree_nodes = static_cast<uint8_t>(args.i32("tree-nodes", 0));
    config.seed = args.has("seed") ? args.u64("seed", 0) : default_seed();
    // 续弈策略：只认 uniform / greedy，未知值直接报错（不做静默回退——一个拼错的策略名
    // 会让整批数据来自另一个教师，而它看起来是"跑完了"）。
    const std::string rollout_policy = args.str("rollout-policy", "uniform");
    if (rollout_policy != "uniform" && rollout_policy != "greedy") {
        throw std::runtime_error("未知 --rollout-policy: " + rollout_policy + "（可选 uniform / greedy）");
    }
    config.greedy_rollout = rollout_policy == "greedy";
    config.shard_samples = args.u64("shard-samples", 100000);
    config.workers = static_cast<uint32_t>(args.i32("workers", 0));
    config.batch_id = args.str("batch-id", make_batch_id(config.policy, config.simulations));
    config.model_id = args.str("model-id", "");
    config.data_dir = args.str("out", default_data_dir(config.batch_id));
    config.control_path = args.str("control", "");
    config.resume = args.flag("continue");
    config.progress_json = args.flag("progress-json");

    // 原样记录命令行参数（输出目录除外，它由 manifest 的地点本身表达）：
    // 数据来源必须能从数据自身读出来，而不是靠翻命令行历史。
    // 续跑时这些参数会与清单里的记录逐项比对（只比对协议项，见 dataio/shard.cpp）。
    //
    // 键名统一成下划线：参数**名**是清单这份数据格式的词汇表（kProtocolParams 与
    // Python 的 SOURCE_KEY_PARAMS 都按下划线拼写），而命令行用连字符只是 CLI 的写法。
    // 两者混在一起会让同一个参数在 C++ 路径与 NN 路径里叫两个名字，来源身份也对不上。
    for (const auto& item : args.items()) {
        if (item.first == "out" || item.first == "quiet" || item.first == "help") continue;
        std::string key = item.first;
        for (char& c : key) {
            if (c == '-') c = '_';
        }
        config.params.emplace_back(key, item.second);
    }

    if (!args.flag("quiet")) {
        log_status("计划：策略 %s%s，样本 %s，种子 %llu，worker %u，输出 %s%s", config.policy.c_str(),
                   config.policy == "mcts" ? ("（模拟 " + std::to_string(config.simulations) + "）").c_str() : "",
                   config.samples_target == 0 ? "不限量" : std::to_string(config.samples_target).c_str(),
                   static_cast<unsigned long long>(config.seed), config.workers == 0 ?
                       std::max(1u, std::thread::hardware_concurrency()) : config.workers,
                   config.data_dir.c_str(), config.resume ? "（续跑）" : "");
    }
    // 纯 C++ 命令自己接管 SIGINT/SIGTERM：Ctrl-C 不该丢尾部数据。
    Control::install_signal_handlers();

    const GenerateResult result = run_generate(config);
    const double per_second =
        result.seconds > 0 ? static_cast<double>(result.samples) / result.seconds : 0.0;
    log_status("完成：状态 %s，样本 %llu，局数 %u，耗时 %.2f 秒（%.0f 样本/秒）",
               batch_state_name(result.state).c_str(),
               static_cast<unsigned long long>(result.samples), result.episodes, result.seconds,
               per_second);
    // stdout 只放可被脚本消费的结果：manifest 路径。
    std::printf("%s\n", result.manifest_path.c_str());
    return 0;
}

}  // namespace r2048
