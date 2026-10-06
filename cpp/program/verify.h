#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace r2048 {

struct VerifyOptions {
    // 重放校验：用清单里记录的参数重新生成整批数据，逐分片比对哈希。
    // 这是本项目对“数据可信”的核心承诺：跑谱结果不是一次性的黑盒产物，
    // 任何一批数据都能被独立重放并逐字节比对。
    bool replay = true;
    // 重放时故意使用与生成时不同的线程数：若数据布局真的与线程数无关，
    // 哈希必须依然一致，这条检查顺带守住了这个不变式。
    uint32_t replay_workers = 1;
    std::string replay_dir;  // 为空则用 <数据目录>_replay，并在校验后删除
};

struct VerifyReport {
    bool ok = false;
    uint64_t samples = 0;
    uint32_t episodes = 0;
    size_t shards = 0;
    bool replay_checked = false;
    double seconds = 0.0;
    std::vector<std::string> problems;
};

VerifyReport verify_batch(const std::string& data_dir, const VerifyOptions& options);

}  // namespace r2048
