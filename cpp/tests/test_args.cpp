#include <cstring>
#include <string>
#include <vector>

#include "core/args.h"
#include "test_util.h"

namespace r2048::tests {

namespace {

// 把 token 列表变成 argc/argv 形式，方便逐条构造用例。
class Argv {
public:
    explicit Argv(std::vector<std::string> tokens) : tokens_(std::move(tokens)) {
        for (const std::string& token : tokens_) raw_.push_back(token.c_str());
    }
    int argc() const { return static_cast<int>(raw_.size()); }
    const char* const* argv() const { return raw_.data(); }

private:
    std::vector<std::string> tokens_;
    std::vector<const char*> raw_;
};

}  // namespace

TEST(args_parses_all_forms) {
    // argv[0] 是子命令名（main 会把 argv+1 传进来），解析器跳过它。
    Argv argv({"generate", "--samples", "1000", "--seed=42", "--quiet", "--ratio", "0.5", "--name",
               "batch-1"});
    const Args args(argv.argc(), argv.argv());

    expect_equal_u64(static_cast<uint64_t>(args.i64("samples", -1)), 1000, "空格分隔的值");
    expect_equal_u64(static_cast<uint64_t>(args.u64("seed", 0)), 42, "等号形式的值");
    expect_true(args.flag("quiet"), "开关参数为真");
    expect_true(!args.flag("verbose"), "未出现的开关为假");
    expect_true(args.flag("verbose", false) == false, "未出现的开关取默认值");
    expect_true(args.has("ratio"), "has 能看到取值参数");
    expect_true(args.has("quiet"), "has 能看到开关参数");
    expect_true(!args.has("missing"), "未给出的参数 has 为假");
    expect_contains(args.str("name", ""), "batch-1", "字符串参数");
    expect_contains(args.str("absent", "fallback"), "fallback", "缺省值");

    const double ratio = args.f64("ratio", 0.0);
    expect_true(ratio > 0.49 && ratio < 0.51, "浮点参数解析正确");
}

TEST(args_rejects_unknown_and_malformed) {
    // 未知键必须报错：拼错的参数如果被静默忽略，一次几十分钟的跑谱会白跑。
    Argv argv({"generate", "--samples", "10", "--simulatoins", "64"});
    const Args args(argv.argc(), argv.argv());
    const std::string message = catch_message([&] { args.expect_only({"samples", "seed"}); });
    expect_contains(message, "simulatoins", "未知参数名出现在错误信息里");
    expect_contains(message, "--samples", "错误信息列出可用参数");

    // 开关型参数不接受取值。
    Argv flag_argv({"generate", "--quiet", "3"});
    const Args flag_args(flag_argv.argc(), flag_argv.argv());
    expect_contains(catch_message([&] { (void)flag_args.flag("quiet"); }), "开关",
                    "开关参数带值时报错");

    // 取值型参数缺少取值。
    Argv missing_argv({"generate", "--samples"});
    const Args missing_args(missing_argv.argc(), missing_argv.argv());
    expect_contains(catch_message([&] { (void)missing_args.i64("samples", 0); }), "需要取值",
                    "缺值时给出明确提示");

    // 非整数、非浮点输入。
    Argv bad_argv({"generate", "--samples", "12x"});
    const Args bad_args(bad_argv.argc(), bad_argv.argv());
    expect_contains(catch_message([&] { (void)bad_args.i64("samples", 0); }), "不是合法整数",
                    "非法整数报错");

    // 位置参数一律拒绝：所有输入都必须具名，否则无法与子命令名区分。
    Argv positional_argv({"generate", "extra"});
    expect_contains(catch_message([&] { (void)Args(positional_argv.argc(), positional_argv.argv()); }),
                    "无法识别的参数", "位置参数被拒绝");
}

}  // namespace r2048::tests
