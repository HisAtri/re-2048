#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "test_util.h"

namespace r2048::tests {

namespace {

bool starts_with(const std::string& text, const std::string& prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

}  // namespace

}  // namespace r2048::tests

int main(int argc, char** argv) {
    using namespace r2048::tests;
    std::vector<std::string> args;
    for (int i = 1; i < argc; i++) args.emplace_back(argv[i]);

    if (!args.empty() && args[0] == "rules-vectors") {
        if (args.size() != 2) {
            std::fprintf(stderr, "用法: r2048_tests rules-vectors <路径>\n");
            return 2;
        }
        return verify_rule_vectors(args[1]);
    }

    // CTest 每个套件注册一条用例，用前缀过滤即可得到按套件的粒度，
    // 不需要为每个套件编译一个可执行文件。
    const std::string filter = args.empty() ? std::string() : args[0];

    int ran = 0;
    int failed = 0;
    for (const TestCase& test : registry().cases) {
        if (!filter.empty() && !starts_with(test.name, filter)) continue;
        ran++;
        try {
            test.body();
            std::printf("  [ok]   %s\n", test.name);
        } catch (const std::exception& error) {
            failed++;
            std::printf("  [FAIL] %s\n         %s\n", test.name, error.what());
        }
    }

    if (ran == 0) {
        std::fprintf(stderr, "没有匹配 \"%s\" 的测试用例\n", filter.c_str());
        return 2;
    }
    if (failed > 0) {
        std::fprintf(stderr, "%d/%d 个用例失败\n", failed, ran);
        return 1;
    }
    std::printf("全部通过（%d 个用例）\n", ran);
    return 0;
}
