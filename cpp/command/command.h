#pragma once

#include <string>
#include <vector>

namespace r2048 {

// 子命令入口：与 KataGo 一样把 CLI 做成“一个二进制 + 多个子命令”，
// 但不把测试编进发行目标（那是 KataGo 的取舍：为便于分发单文件而牺牲增量构建）。
// 这里的可执行文件只做用户真正会敲的事情。
struct SubCommand {
    const char* name;
    const char* summary;
    int (*run)(int argc, const char* const* argv);
};

const std::vector<SubCommand>& sub_commands();

int run_version(int argc, const char* const* argv);
int run_generate(int argc, const char* const* argv);
int run_evaluate(int argc, const char* const* argv);
int run_verify(int argc, const char* const* argv);
int run_bench(int argc, const char* const* argv);
void print_help();

}  // namespace r2048
