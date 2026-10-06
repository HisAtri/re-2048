#include "core/json.h"

#include <cstdio>
#include <stdexcept>

namespace r2048 {

Json Json::boolean(bool value) {
    Json json;
    json.kind_ = Kind::Bool;
    json.bool_ = value;
    return json;
}

Json Json::integer(int64_t value) {
    Json json;
    json.kind_ = Kind::Int;
    json.int_ = value;
    return json;
}

Json Json::str(std::string value) {
    Json json;
    json.kind_ = Kind::String;
    json.string_ = std::move(value);
    return json;
}

Json Json::array() {
    Json json;
    json.kind_ = Kind::Array;
    return json;
}

Json Json::object() {
    Json json;
    json.kind_ = Kind::Object;
    return json;
}

bool Json::as_bool() const {
    if (kind_ != Kind::Bool) throw std::runtime_error("JSON 值不是布尔");
    return bool_;
}

int64_t Json::as_int() const {
    if (kind_ != Kind::Int) throw std::runtime_error("JSON 值不是整数");
    return int_;
}

const std::string& Json::as_string() const {
    if (kind_ != Kind::String) throw std::runtime_error("JSON 值不是字符串");
    return string_;
}

void Json::set(const std::string& key, Json value) {
    if (kind_ != Kind::Object) throw std::runtime_error("JSON 值不是对象，不能设置键");
    for (auto& entry : object_) {
        if (entry.first == key) {
            entry.second = std::move(value);
            return;
        }
    }
    object_.emplace_back(key, std::move(value));
}

void Json::push(Json value) {
    if (kind_ != Kind::Array) throw std::runtime_error("JSON 值不是数组，不能追加元素");
    array_.push_back(std::move(value));
}

const Json& Json::at(const std::string& key) const {
    if (kind_ != Kind::Object) throw std::runtime_error("JSON 值不是对象");
    for (const auto& entry : object_) {
        if (entry.first == key) return entry.second;
    }
    throw std::runtime_error("JSON 对象缺少键: " + key);
}

const Json& Json::at(size_t index) const {
    if (kind_ != Kind::Array) throw std::runtime_error("JSON 值不是数组");
    if (index >= array_.size()) throw std::runtime_error("JSON 数组下标越界");
    return array_[index];
}

size_t Json::size() const {
    if (kind_ == Kind::Array) return array_.size();
    if (kind_ == Kind::Object) return object_.size();
    return 0;
}

bool Json::has(const std::string& key) const {
    if (kind_ != Kind::Object) return false;
    for (const auto& entry : object_) {
        if (entry.first == key) return true;
    }
    return false;
}

const std::vector<std::pair<std::string, Json>>& Json::items() const {
    if (kind_ != Kind::Object) throw std::runtime_error("JSON 值不是对象");
    return object_;
}

const std::vector<Json>& Json::elements() const {
    if (kind_ != Kind::Array) throw std::runtime_error("JSON 值不是数组");
    return array_;
}

std::string json_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (const char raw : text) {
        const unsigned char ch = static_cast<unsigned char>(raw);
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (ch < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", ch);
                    out += buffer;
                } else {
                    // 其余字节原样输出：我们的字符串都是 UTF-8，JSON 允许直接写。
                    out += raw;
                }
        }
    }
    return out;
}

void Json::dump_to(std::string& out, int indent, int depth) const {
    const auto newline_indent = [&](int level) {
        if (indent <= 0) return;
        out += '\n';
        out.append(static_cast<size_t>(indent * level), ' ');
    };

    switch (kind_) {
        case Kind::Null:
            out += "null";
            break;
        case Kind::Bool:
            out += bool_ ? "true" : "false";
            break;
        case Kind::Int:
            out += std::to_string(int_);
            break;
        case Kind::String:
            out += '"';
            out += json_escape(string_);
            out += '"';
            break;
        case Kind::Array: {
            if (array_.empty()) {
                out += "[]";
                break;
            }
            out += '[';
            for (size_t i = 0; i < array_.size(); i++) {
                if (i > 0) out += ',';
                newline_indent(depth + 1);
                array_[i].dump_to(out, indent, depth + 1);
            }
            newline_indent(depth);
            out += ']';
            break;
        }
        case Kind::Object: {
            if (object_.empty()) {
                out += "{}";
                break;
            }
            out += '{';
            for (size_t i = 0; i < object_.size(); i++) {
                if (i > 0) out += ',';
                newline_indent(depth + 1);
                out += '"';
                out += json_escape(object_[i].first);
                out += "\":";
                if (indent > 0) out += ' ';
                object_[i].second.dump_to(out, indent, depth + 1);
            }
            newline_indent(depth);
            out += '}';
            break;
        }
    }
}

std::string Json::dump(int indent) const {
    std::string out;
    dump_to(out, indent, 0);
    if (indent > 0) out += '\n';
    return out;
}

namespace {

class Parser {
public:
    explicit Parser(const std::string& text) : text_(text) {}

    Json parse() {
        skip_whitespace();
        Json value = parse_value();
        skip_whitespace();
        if (position_ != text_.size()) fail("末尾存在多余内容");
        return value;
    }

private:
    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error("JSON 解析失败（位置 " + std::to_string(position_) + "）: " + message);
    }

    void skip_whitespace() {
        while (position_ < text_.size()) {
            const char ch = text_[position_];
            if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
                position_++;
            } else {
                break;
            }
        }
    }

    char peek() const {
        if (position_ >= text_.size()) throw std::runtime_error("JSON 解析失败: 输入意外结束");
        return text_[position_];
    }

    void expect(char expected) {
        if (position_ >= text_.size() || text_[position_] != expected) {
            fail(std::string("期望字符 '") + expected + "'");
        }
        position_++;
    }

    bool consume_literal(const char* literal) {
        const size_t length = std::char_traits<char>::length(literal);
        if (text_.compare(position_, length, literal) != 0) return false;
        position_ += length;
        return true;
    }

    Json parse_value() {
        if (position_ >= text_.size()) fail("缺少值");
        switch (text_[position_]) {
            case '{': return parse_object();
            case '[': return parse_array();
            case '"': return Json::str(parse_string());
            case 't':
                if (consume_literal("true")) return Json::boolean(true);
                fail("非法字面量");
            case 'f':
                if (consume_literal("false")) return Json::boolean(false);
                fail("非法字面量");
            case 'n':
                if (consume_literal("null")) return Json();
                fail("非法字面量");
            default: return parse_number();
        }
    }

    Json parse_object() {
        expect('{');
        Json object = Json::object();
        skip_whitespace();
        if (peek() == '}') {
            position_++;
            return object;
        }
        while (true) {
            skip_whitespace();
            const std::string key = parse_string();
            skip_whitespace();
            expect(':');
            skip_whitespace();
            object.set(key, parse_value());
            skip_whitespace();
            const char next = peek();
            if (next == ',') {
                position_++;
                continue;
            }
            if (next == '}') {
                position_++;
                return object;
            }
            fail("对象中期望 ',' 或 '}'");
        }
    }

    Json parse_array() {
        expect('[');
        Json array = Json::array();
        skip_whitespace();
        if (peek() == ']') {
            position_++;
            return array;
        }
        while (true) {
            skip_whitespace();
            array.push(parse_value());
            skip_whitespace();
            const char next = peek();
            if (next == ',') {
                position_++;
                continue;
            }
            if (next == ']') {
                position_++;
                return array;
            }
            fail("数组中期望 ',' 或 ']'");
        }
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            if (position_ >= text_.size()) fail("字符串未闭合");
            const char ch = text_[position_++];
            if (ch == '"') return out;
            if (ch != '\\') {
                out += ch;
                continue;
            }
            if (position_ >= text_.size()) fail("转义序列未结束");
            const char escape = text_[position_++];
            switch (escape) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    // 只支持 \uXXXX：解成 UTF-8。代理对不属于本项目的使用范围，
                    // 遇到就报错而不是猜（写入侧从不产生代理对）。
                    if (position_ + 4 > text_.size()) fail("\\u 转义不完整");
                    uint32_t code = 0;
                    for (int i = 0; i < 4; i++) {
                        const char hex = text_[position_++];
                        code <<= 4;
                        if (hex >= '0' && hex <= '9') {
                            code |= static_cast<uint32_t>(hex - '0');
                        } else if (hex >= 'a' && hex <= 'f') {
                            code |= static_cast<uint32_t>(hex - 'a' + 10);
                        } else if (hex >= 'A' && hex <= 'F') {
                            code |= static_cast<uint32_t>(hex - 'A' + 10);
                        } else {
                            fail("\\u 转义含非法十六进制字符");
                        }
                    }
                    if (code < 0x80) {
                        out += static_cast<char>(code);
                    } else if (code < 0x800) {
                        out += static_cast<char>(0xC0 | (code >> 6));
                        out += static_cast<char>(0x80 | (code & 0x3F));
                    } else {
                        out += static_cast<char>(0xE0 | (code >> 12));
                        out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                        out += static_cast<char>(0x80 | (code & 0x3F));
                    }
                    break;
                }
                default: fail("未知的转义字符");
            }
        }
    }

    Json parse_number() {
        const size_t start = position_;
        if (position_ < text_.size() && (text_[position_] == '-' || text_[position_] == '+')) position_++;
        while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') position_++;
        if (position_ == start) fail("非法值（本项目只使用整数，不支持浮点）");
        if (position_ < text_.size() && (text_[position_] == '.' || text_[position_] == 'e' ||
                                         text_[position_] == 'E')) {
            fail("不支持浮点数");
        }
        int64_t value = 0;
        try {
            value = std::stoll(text_.substr(start, position_ - start));
        } catch (const std::exception&) {
            fail("整数超出范围");
        }
        return Json::integer(value);
    }

    const std::string& text_;
    size_t position_ = 0;
};

}  // namespace

Json parse_json(const std::string& text) { return Parser(text).parse(); }

}  // namespace r2048
