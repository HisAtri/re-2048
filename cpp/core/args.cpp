#include "core/args.h"

#include <charconv>
#include <stdexcept>

namespace r2048 {

Args::Args(int argc, const char* const* argv) {
    for (int i = 1; i < argc; i++) {
        const std::string token = argv[i];
        if (token.size() < 3 || token[0] != '-' || token[1] != '-') {
            throw std::runtime_error("无法识别的参数: " + token + "（只接受 --key value / --key=value / --flag）");
        }
        std::string key = token.substr(2);
        std::string value;
        bool has_value = false;
        const size_t eq = key.find('=');
        if (eq != std::string::npos) {
            value = key.substr(eq + 1);
            key = key.substr(0, eq);
            has_value = true;
        } else if (i + 1 < argc && argv[i + 1][0] != '-') {
            // 下一个 token 不以 '-' 开头就当成本参数的值。没有位置参数，因此不存在歧义。
            value = argv[++i];
            has_value = true;
        }
        if (key.empty()) throw std::runtime_error("空的参数名: " + token);
        if (has_value) {
            values_[key] = value;
        } else {
            flags_.insert(key);
        }
    }
}

bool Args::has(const std::string& key) const {
    return values_.count(key) > 0 || flags_.count(key) > 0;
}

std::string Args::require_value(const std::string& key) const {
    const auto it = values_.find(key);
    if (it != values_.end()) return it->second;
    if (flags_.count(key) > 0) throw std::runtime_error("参数 --" + key + " 需要取值");
    throw std::runtime_error("缺少必需参数 --" + key);
}

std::string Args::str(const std::string& key, const std::string& def) const {
    if (!has(key)) return def;
    return require_value(key);
}

int64_t Args::i64(const std::string& key, int64_t def) const {
    if (!has(key)) return def;
    const std::string text = require_value(key);
    int64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc() || result.ptr != text.data() + text.size()) {
        throw std::runtime_error("参数 --" + key + " 不是合法整数: " + text);
    }
    return value;
}

uint64_t Args::u64(const std::string& key, uint64_t def) const {
    if (!has(key)) return def;
    const std::string text = require_value(key);
    uint64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc() || result.ptr != text.data() + text.size()) {
        throw std::runtime_error("参数 --" + key + " 不是合法非负整数: " + text);
    }
    return value;
}

int32_t Args::i32(const std::string& key, int32_t def) const {
    const int64_t value = i64(key, def);
    if (value < INT32_MIN || value > INT32_MAX) {
        throw std::runtime_error("参数 --" + key + " 超出 32 位整数范围: " + std::to_string(value));
    }
    return static_cast<int32_t>(value);
}

double Args::f64(const std::string& key, double def) const {
    if (!has(key)) return def;
    const std::string text = require_value(key);
    size_t consumed = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error("参数 --" + key + " 不是合法浮点数: " + text);
    }
    if (consumed != text.size()) {
        throw std::runtime_error("参数 --" + key + " 不是合法浮点数: " + text);
    }
    return value;
}

bool Args::flag(const std::string& key) const {
    // 与带默认值的重载保持一致：`--quiet 3` 被解析成“取值”之后，必须在这里
    // 报错，而不是返回 false 让调用方以为开关没打开。
    if (values_.count(key) > 0) throw std::runtime_error("参数 --" + key + " 是开关，不接受取值");
    return flags_.count(key) > 0;
}

bool Args::flag(const std::string& key, bool def) const {
    if (values_.count(key) > 0) throw std::runtime_error("参数 --" + key + " 是开关，不接受取值");
    if (flags_.count(key) > 0) return true;
    return def;
}

void Args::expect_only(const std::vector<std::string>& known) const {
    std::set<std::string> allowed(known.begin(), known.end());
    std::string unknown;
    for (const auto& entry : values_) {
        if (allowed.count(entry.first) == 0) unknown += (unknown.empty() ? "" : ", ") + entry.first;
    }
    for (const auto& key : flags_) {
        if (allowed.count(key) == 0) unknown += (unknown.empty() ? "" : ", ") + key;
    }
    if (unknown.empty()) return;

    std::string list;
    for (const auto& key : known) list += (list.empty() ? "" : ", ") + ("--" + key);
    throw std::runtime_error("不认识的参数: " + unknown + "\n可用参数: " + list);
}

std::vector<std::pair<std::string, std::string>> Args::items() const {
    std::vector<std::pair<std::string, std::string>> out;
    out.reserve(values_.size() + flags_.size());
    for (const auto& entry : values_) out.emplace_back(entry.first, entry.second);
    for (const auto& key : flags_) out.emplace_back(key, "true");
    return out;
}

}  // namespace r2048
