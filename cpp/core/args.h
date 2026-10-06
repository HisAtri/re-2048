#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace r2048 {

// 命令行解析器：只接受 --key value、--key=value、--flag 三种写法，不接受位置参数。
//
// 最值钱的行为是 expect_only()：每个子命令声明自己认识的键，出现未知键立即报错。
// 静默忽略拼错的参数会让一次几十分钟的跑谱白跑，或者更糟——用默认值跑出一批
// 语义不对的数据。这里宁可直接失败。
class Args {
public:
    Args(int argc, const char* const* argv);

    bool has(const std::string& key) const;

    std::string str(const std::string& key, const std::string& def) const;
    int64_t i64(const std::string& key, int64_t def) const;
    uint64_t u64(const std::string& key, uint64_t def) const;
    int32_t i32(const std::string& key, int32_t def) const;
    double f64(const std::string& key, double def) const;

    // 开关型参数：出现即为 true。
    bool flag(const std::string& key) const;
    // 开关型参数，未出现时为 def，用于 --no-xxx 这类写法。
    bool flag(const std::string& key, bool def) const;

    // 校验：除 known 之外的键都会触发异常。--- help 由调用方自行处理。
    void expect_only(const std::vector<std::string>& known) const;

    // 全部已给出的参数，按 key 排序。用于写入产物的来源记录。
    std::vector<std::pair<std::string, std::string>> items() const;

private:
    std::string require_value(const std::string& key) const;

    std::map<std::string, std::string> values_;
    std::set<std::string> flags_;
};

}  // namespace r2048
