#pragma once

#include <cstdint>

namespace r2048 {

// xoshiro256**：整数运算完全定义，跨平台、跨编译器结果一致。
//
// 跑谱的随机流必须能按种子复现，而且并行 lane 数量、线程调度顺序都不允许影响
// 任何一局的结果。因此这里不用 std::random_device，也不共享全局发生器：
// 每个 lane / 每局持有独立实例，种子由上层按 (episode, step, ...) 确定性派生。
// 也正因如此，选择算法时要的是“规格明确”而不是“最快”：std::mt19937_64 同样
// 规格明确，但 xoshiro256** 在这个尺寸上更快且状态更小。
class Rng {
public:
    explicit Rng(uint64_t seed) { reseed(seed); }

    void reseed(uint64_t seed) {
        // splitmix64 展开种子：base + episode 这类相邻种子必须产生不相关的流，
        // 否则不同局会意外共享随机序列。
        uint64_t z = seed;
        for (int i = 0; i < 4; i++) {
            z += 0x9e3779b97f4a7c15ull;
            uint64_t x = z;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
            state_[i] = x ^ (x >> 31);
        }
    }

    uint64_t next_u64() {
        const uint64_t result = rotl(state_[1] * 5, 7) * 9;
        const uint64_t t = state_[1] << 17;
        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= t;
        state_[3] = rotl(state_[3], 45);
        return result;
    }

    // [0, n) 上的均匀整数。用高 32 位乘法取整，偏差量级 2^-32，对 16 个空格选位
    // 与 90/10 判定完全够用；刻意不做拒绝采样，避免让随机流依赖具体 n。
    uint32_t below(uint32_t n) {
        return n == 0 ? 0 : static_cast<uint32_t>((next_u64() >> 32) * static_cast<uint64_t>(n) >> 32);
    }

    // 十分之一的概率为真。用取模而不是浮点比较：浮点在不同 -ffast-math 设置下
    // 结果可能不同，取模是整数运算，确定性不依赖编译标志。
    bool one_in_ten() { return next_u64() % 10 == 0; }

private:
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

    uint64_t state_[4];
};

}  // namespace r2048
