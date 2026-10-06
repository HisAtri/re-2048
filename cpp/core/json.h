#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace r2048 {

// 极小 JSON 支持：只为 manifest 这类“C++ 写、C++ 与 Python 都要读”的文件服务。
//
// 为什么自己写而不是引第三方：C++ 侧保持零第三方依赖是本项目的硬约束（旧版把
// 这条当作架构理由）。为什么不用手写字符串拼接 + 线性查找：那样 writer 与 reader
// 会各自维护一份对格式的假设，任一侧改动就可能悄悄错位。这里实现一个**受限但完整**
// 的 JSON 子集：对象、数组、字符串、整数、布尔、null；不支持浮点与 \u 转义之外的
// 花式写法，遇到就报错而不是猜。
class Json {
public:
    enum class Kind { Null, Bool, Int, String, Array, Object };

    Json() : kind_(Kind::Null) {}
    static Json boolean(bool value);
    static Json integer(int64_t value);
    static Json str(std::string value);
    static Json array();
    static Json object();

    Kind kind() const { return kind_; }
    bool is_null() const { return kind_ == Kind::Null; }
    bool is_object() const { return kind_ == Kind::Object; }
    bool is_array() const { return kind_ == Kind::Array; }

    bool as_bool() const;
    int64_t as_int() const;
    const std::string& as_string() const;

    // 对象/数组访问。类型不符或键不存在时抛异常：manifest 缺字段就是数据损坏，
    // 不应该被当成“用默认值继续跑”。
    void set(const std::string& key, Json value);
    void push(Json value);
    const Json& at(const std::string& key) const;
    const Json& at(size_t index) const;
    size_t size() const;
    bool has(const std::string& key) const;
    const std::vector<std::pair<std::string, Json>>& items() const;
    const std::vector<Json>& elements() const;

    std::string dump(int indent = 0) const;

private:
    void dump_to(std::string& out, int indent, int depth) const;

    Kind kind_;
    bool bool_ = false;
    int64_t int_ = 0;
    std::string string_;
    std::vector<Json> array_;
    std::vector<std::pair<std::string, Json>> object_;
};

Json parse_json(const std::string& text);
std::string json_escape(const std::string& text);

}  // namespace r2048
