#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "game/rules.h"
#include "test_util.h"

namespace r2048::tests {

namespace {

// 跨语言一致性向量文件格式（小端）：
//   magic "R2048VEC" (8 字节) | u32 version | u32 count
//   每条记录：board[16] + 4 × { afterstate[16], u64 score, u8 valid }   // 方向顺序 0..3
// 由 Python 参考实现生成（python/tests/test_rules_conformance.py），C++ 侧逐条比对。
constexpr size_t kRecordBytes = 16 + 4 * (16 + 8 + 1);

uint32_t read_u32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t read_u64(const uint8_t* p) {
    uint64_t value = 0;
    for (int i = 7; i >= 0; i--) value = (value << 8) | p[i];
    return value;
}

std::vector<uint8_t> read_all(const std::string& path) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) fail("无法打开向量文件: " + path);
    std::vector<uint8_t> data;
    uint8_t buffer[1 << 16];
    size_t read = 0;
    while ((read = std::fread(buffer, 1, sizeof(buffer), file)) > 0) {
        data.insert(data.end(), buffer, buffer + read);
    }
    const bool failed = std::ferror(file) != 0;
    std::fclose(file);
    if (failed) fail("读取向量文件失败: " + path);
    return data;
}

}  // namespace

int verify_rule_vectors(const std::string& path) {
    const std::vector<uint8_t> data = read_all(path);
    if (data.size() < 16 || std::memcmp(data.data(), "R2048VEC", 8) != 0) {
        std::fprintf(stderr, "向量文件头不合法: %s\n", path.c_str());
        return 2;
    }
    const uint32_t version = read_u32(data.data() + 8);
    const uint32_t count = read_u32(data.data() + 12);
    if (version != 1) {
        std::fprintf(stderr, "向量文件版本不支持: %u\n", version);
        return 2;
    }
    const size_t expected_size = 16 + kRecordBytes * static_cast<size_t>(count);
    if (data.size() != expected_size) {
        std::fprintf(stderr, "向量文件长度不符: 实际 %zu，期望 %zu（%u 条）\n", data.size(),
                     expected_size, count);
        return 2;
    }

    for (uint32_t index = 0; index < count; index++) {
        const uint8_t* record = data.data() + 16 + kRecordBytes * index;
        Board board;
        std::memcpy(board.tiles, record, 16);
        const uint8_t* cursor = record + 16;
        for (int action = 0; action < kNumActions; action++) {
            Board expected_board;
            std::memcpy(expected_board.tiles, cursor, 16);
            const uint64_t expected_score = read_u64(cursor + 16);
            const bool expected_valid = cursor[24] != 0;
            cursor += 25;

            const MoveOutcome actual = apply_move(board, action);
            const bool mismatch = actual.board != expected_board || actual.score != expected_score ||
                                  actual.valid != expected_valid;
            if (mismatch) {
                std::fprintf(stderr,
                             "第 %u 条记录方向 %d 不一致\n  棋盘 %s\n  Python: %s 分 %llu %s\n  C++   : %s 分 %llu %s\n",
                             index, action, board.to_string().c_str(), expected_board.to_string().c_str(),
                             static_cast<unsigned long long>(expected_score),
                             expected_valid ? "有效" : "无效", actual.board.to_string().c_str(),
                             static_cast<unsigned long long>(actual.score),
                             actual.valid ? "有效" : "无效");
                return 1;
            }
        }
    }
    std::printf("跨语言规则一致性通过（%u 条向量）\n", count);
    return 0;
}

}  // namespace r2048::tests
