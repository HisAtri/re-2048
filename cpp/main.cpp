#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "command/command.h"
#include "core/json.h"
#include "core/logging.h"
#include "core/version.h"
#include "r2048_build_id.h"  // 由 CMake 生成，见 cmake/R2048BuildId.cmake

namespace r2048 {

const std::vector<SubCommand>& sub_commands() {
    static const std::vector<SubCommand> commands = {
        {"generate", "跑谱：自对弈生成训练数据（random / mcts）", run_generate},
        {"evaluate", "评测：跑固定局数并汇总成绩", run_evaluate},
        {"verify", "校验数据批：摘要、结构，并重放比对哈希", run_verify},
        {"bench", "微基准：移动生成 / 随机对局 / UCT-rollout 吞吐", run_bench},
        {"version", "打印版本与构建身份", run_version},
    };
    return commands;
}

int run_version(int argc, const char* const* argv) {
    (void)argc;
    (void)argv;
    Json info = Json::object();
    info.set("rules_version", Json::integer(kRulesVersion));
    info.set("protocol_version", Json::integer(kProtocolVersion));
    info.set("build_id", Json::str(R2048_BUILD_ID));
    info.set("git", Json::str(R2048_BUILD_GIT_DESC));
    info.set("build_type", Json::str(R2048_BUILD_TYPE));
    std::printf("%s\n", info.dump(0).c_str());
    return 0;
}

void print_help() {
    std::printf("用法: r2048 <子命令> [--key value ...]\n\n子命令:\n");
    for (const SubCommand& command : sub_commands()) {
        std::printf("  %-10s %s\n", command.name, command.summary);
    }
    std::printf("\n所有参数都是具名的（--key value 或 --key=value）；未知参数会直接报错。\n");
}

}  // namespace r2048

int main(int argc, char** argv) {
    using namespace r2048;
    if (argc < 2) {
        print_help();
        return 1;
    }
    const std::string name = argv[1];
    if (name == "--help" || name == "-h" || name == "help") {
        print_help();
        return 0;
    }

    try {
        for (const SubCommand& command : sub_commands()) {
            if (name == command.name) return command.run(argc - 1, argv + 1);
        }
        log_error("未知子命令: %s", name.c_str());
        print_help();
        return 1;
    } catch (const std::exception& error) {
        log_error("%s", error.what());
        return 1;
    }
}
