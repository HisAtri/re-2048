#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace r2048 {

// 棋盘哈希：把 16 个字节混合成 64 位。它只用于缓存/置换表的索引，
// 命中时一律再比一遍完整棋盘，因此这里要的是分布好、延迟低，而不是抗碰撞。
inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

inline uint64_t board_hash(const uint8_t* tiles) {
    uint64_t lo = 0;
    uint64_t hi = 0;
    __builtin_memcpy(&lo, tiles, 8);
    __builtin_memcpy(&hi, tiles + 8, 8);
    const uint64_t a = mix64(lo ^ 0x9e3779b97f4a7c15ull);
    const uint64_t b = mix64(hi ^ 0xc2b2ae3d27d4eb4full);
    // 最后再混一次：只用 a ^ rotl(b) 时低位分布明显偏斜（对稀疏棋盘尤其明显），
    // 而缓存索引恰好用的是低位。实测过后果：几个高频局面反复落在同一个 set 里
    // 互相驱逐，推理缓存命中率崩塌，引擎陷入“永远在补发同一批请求”的活锁。
    return mix64(a ^ ((b << 32) | (b >> 32)));
}

// SHA-256：只用于分片与 manifest 的自校验，不在热路径上。自己实现是为了让
// C++ 侧保持零第三方依赖（旧版把这条当作硬约束，代价可控而收益明确）。
class Sha256 {
public:
    Sha256();
    void update(const void* data, size_t len);
    // 结束并返回 32 字节摘要；此后不可再 update。
    void finish(uint8_t out[32]);

private:
    void transform(const uint8_t block[64]);

    uint32_t state_[8];
    uint64_t bit_count_;
    uint8_t buffer_[64];
    size_t buffer_len_;
};

std::string sha256_hex(const void* data, size_t len);
std::string sha256_hex_file(const std::string& path);
std::string to_hex(const uint8_t* bytes, size_t len);

}  // namespace r2048
