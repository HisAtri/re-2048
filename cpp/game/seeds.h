#pragma once

#include <cstdint>

#include "core/hash.h"

namespace r2048 {

// 随机流的用途标签。同一局里“环境生成块”和“动作采样”必须用互相独立的流，
// 否则改一处采样策略会连带改变另一处的随机序列，实验就无法单独对比。
enum SeedSalt : uint64_t {
    kSaltEnvironment = 0x1,  // 环境：生成新块的位置与数值
    kSaltAction = 0x2,       // 动作采样：按温度 / 探索分布选方向
    kSaltRollout = 0x3,      // 随机 rollout
    kSaltShuffle = 0x4,      // 数据打乱与采样
};

// 种子派生：每个随机流都从 (基础种子, 用途, 局号, 步号, 序号) 直接算出，
// 而不是从一条全局流里顺序取。
//
// 这样做的直接好处：并行 lane 数量、线程调度顺序都不影响任何一局的结果，
// 而且任意一局任意一步都能被单独重放到相同的随机序列 —— 这是 verify 子命令
// 能对跑谱结果做重放校验的前提。
inline uint64_t derive_seed(uint64_t base_seed, SeedSalt salt, uint64_t episode, uint64_t step,
                            uint64_t index) {
    uint64_t h = mix64(base_seed ^ (static_cast<uint64_t>(salt) * 0x9e3779b97f4a7c15ull));
    h = mix64(h ^ (episode + 0x165667b19e3779f9ull));
    h = mix64(h ^ (step * 0x27d4eb2f165667c5ull));
    h = mix64(h ^ (index * 0x85ebca77c2b2ae63ull));
    return h;
}

}  // namespace r2048
