#include <cstdio>
#include <stdexcept>

#include "command/command.h"
#include "core/args.h"
#include "core/json.h"
#include "core/logging.h"
#include "program/verify.h"

namespace r2048 {

namespace {

void print_usage() {
    std::printf(
        "用法: r2048 verify --data <数据目录> [选项]\n\n"
        "校验一个数据批：清单与分片的一致性、逐分片摘要、记录内局号的单调性，\n"
        "并（默认）用清单记录的参数重新生成整批数据逐分片比对哈希。\n"
        "重放会使用与生成时不同的线程数，因此它同时验证了“数据布局与线程数无关”。\n\n"
        "  --data            数据批目录（必需）\n"
        "  --no-replay       只做结构与摘要校验，不重放\n"
        "  --replay-workers  重放时的 worker 数（默认 1）\n"
        "  --replay-dir      重放的临时目录（默认 <数据目录>_replay，校验后删除）\n");
}

}  // namespace

int run_verify(int argc, const char* const* argv) {
    Args args(argc, argv);
    if (args.flag("help")) {
        print_usage();
        return 0;
    }
    args.expect_only({"data", "no-replay", "replay-workers", "replay-dir", "help"});

    VerifyOptions options;
    options.replay = !args.flag("no-replay");
    options.replay_workers = static_cast<uint32_t>(args.i32("replay-workers", 1));
    options.replay_dir = args.str("replay-dir", "");

    const std::string data_dir = args.str("data", "");
    if (data_dir.empty()) {
        print_usage();
        throw std::runtime_error("必须指定 --data");
    }

    log_status("校验 %s（%s）", data_dir.c_str(), options.replay ? "含重放" : "仅结构与摘要");
    const VerifyReport report = verify_batch(data_dir, options);

    Json summary = Json::object();
    summary.set("data", Json::str(data_dir));
    summary.set("ok", Json::boolean(report.ok));
    summary.set("samples", Json::integer(static_cast<int64_t>(report.samples)));
    summary.set("episodes", Json::integer(report.episodes));
    summary.set("shards", Json::integer(static_cast<int64_t>(report.shards)));
    summary.set("replay_checked", Json::boolean(report.replay_checked));
    summary.set("seconds", Json::integer(static_cast<int64_t>(report.seconds + 0.5)));
    Json problems = Json::array();
    for (const std::string& problem : report.problems) problems.push(Json::str(problem));
    summary.set("problems", std::move(problems));

    for (const std::string& problem : report.problems) log_error("%s", problem.c_str());
    if (report.ok) {
        log_status("校验通过：样本 %llu，局数 %u，分片 %zu，耗时 %.1f 秒%s",
                   static_cast<unsigned long long>(report.samples), report.episodes, report.shards,
                   report.seconds, report.replay_checked ? "（已重放比对哈希）" : "");
    } else {
        log_status("校验失败：%zu 个问题，耗时 %.1f 秒", report.problems.size(), report.seconds);
    }
    std::printf("%s\n", summary.dump(0).c_str());
    return report.ok ? 0 : 1;
}

}  // namespace r2048
