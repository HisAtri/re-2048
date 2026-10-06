#pragma once

#include <cstdint>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace r2048::tests {

// 极简测试设施：不引入测试框架。我们要守的是少量高价值不变量
// （规则一致性、确定性、数据往返、协议契约），断言失败时能打印出
// 具体是哪一块棋盘、哪个方向错就够了，这不需要一个依赖。
struct TestCase {
    const char* name;
    std::function<void()> body;
};

struct Registry {
    std::vector<TestCase> cases;
};

inline Registry& registry() {
    static Registry instance;
    return instance;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> body) { registry().cases.push_back({name, body}); }
};

// 断言失败抛出带上下文的消息，由 run_suite 统一打印。
inline void fail(const std::string& message) { throw std::runtime_error(message); }

inline void expect_true(bool condition, const std::string& what) {
    if (!condition) fail("断言失败: " + what);
}

inline void expect_equal_u64(uint64_t actual, uint64_t expected, const std::string& what) {
    if (actual != expected) {
        fail("断言失败: " + what + "（实际 " + std::to_string(actual) + "，期望 " +
             std::to_string(expected) + "）");
    }
}

inline void expect_contains(const std::string& haystack, const std::string& needle,
                            const std::string& what) {
    if (haystack.find(needle) == std::string::npos) {
        fail("断言失败: " + what + "（未包含 \"" + needle + "\"，实际为 \"" + haystack + "\"）");
    }
}

// 规则一致性向量的校验入口：读取 Python 参考实现生成的期望结果并与 C++ 实现比对。
// 这是唯一一处“另一门语言算出来的期望值”，跨语言契约靠它把关。
int verify_rule_vectors(const std::string& path);

// 捕获异常并返回其消息：用于断言“错误信息里必须包含什么”。
// 直接失败（而不是期望异常）会让“本该报错却静默通过”的用例暴露出来。
inline std::string catch_message(const std::function<void()>& body) {
    try {
        body();
    } catch (const std::exception& error) {
        return error.what();
    }
    return "<没有抛出异常>";
}

// 把字符串列表拼成一段文本，便于断言错误集合的内容。
inline std::string join_text(const std::vector<std::string>& items) {
    std::string out;
    for (const std::string& item : items) {
        if (!out.empty()) out += " | ";
        out += item;
    }
    return out;
}

}  // namespace r2048::tests

// 用例注册宏：在测试文件里写 TEST(name) { ... } 即可。
#define TEST(test_name)                                                                       \
    static void test_body_##test_name();                                                      \
    static ::r2048::tests::Registrar registrar_##test_name(#test_name, test_body_##test_name); \
    static void test_body_##test_name()
