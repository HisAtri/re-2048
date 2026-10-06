#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "core/fileio.h"
#include "dataio/shard.h"
#include "search/batched_engine.h"
#include "test_util.h"

namespace r2048::tests {

namespace {

// 用合成结果驱动批推理引擎：value 全 0、logits 全 0（等价于均匀先验）。
// 这样可以在完全不含 Python 与神经网络的情况下验证两阶段协议、写入顺序与终止条件，
// 也便于在引擎出错时用调试器直接定位，而不必穿过 pybind 与 torch。
struct SyntheticDriver {
    BatchedEngine& engine;
    uint64_t rounds = 0;
    uint64_t requests = 0;

    bool run_until_done(uint64_t max_rounds = 200000) {
        std::vector<uint64_t> ids;
        std::vector<float> values;
        std::vector<float> logits;
        while (rounds < max_rounds) {
            rounds++;
            BatchedEngine::Advance advance = engine.advance(4096);
            const size_t count = advance.request_ids.size();
            if (count > 0) {
                requests += count;
                values.assign(count, 0.0f);
                logits.assign(count * kNumActions, 0.0f);
                engine.submit(advance.request_ids.data(), values.data(), logits.data(), count);
            } else if (advance.done) {
                return true;
            } else {
                // 既没有请求也没有完成：引擎不可能自行前进，直接判定卡死。
                return false;
            }
        }
        return false;
    }
};

class TempDir {
public:
    explicit TempDir(const std::string& name) {
        path_ = (std::filesystem::temp_directory_path() / ("r2048-engine-" + name)).string();
        std::filesystem::remove_all(path_);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    std::string sub(const std::string& name) const { return path_ + "/" + name; }

private:
    std::string path_;
};

}  // namespace

TEST(engine_completes_and_writes_in_episode_order) {
    TempDir dir("basic");
    EngineConfig config;
    config.lanes = 8;
    config.simulations = 16;
    config.in_flight_per_lane = 4;
    config.cache_entries = 1 << 12;
    config.seed = 20261006;

    BatchedEngine engine(config, dir.sub("batch"), "engine-test", "synthetic", 3000, 1000);
    SyntheticDriver driver{engine};
    const bool done = driver.run_until_done();
    expect_true(done, "引擎应当在有限轮次内达到目标样本数");

    const std::string manifest_path = engine.finish();
    const BatchManifest manifest = read_manifest(dir.sub("batch"));
    expect_equal_u64(manifest.samples, 3000, "样本数恰好等于目标");
    expect_true(manifest.meta.policy == "nn", "策略标记为 nn");
    expect_true(manifest.meta.model_id == "synthetic", "来源模型记录在清单里");
    expect_equal_u64(manifest.episodes, manifest.shards.back().last_episode + 1,
                     "局数与末分片覆盖的局号一致");
    expect_contains(manifest_path, "manifest.json", "返回 manifest 路径");

    const EngineStats stats = engine.stats();
    expect_true(stats.requests_issued > 0, "确实发出过批推理请求");
    expect_equal_u64(stats.requests_issued, stats.requests_submitted, "请求与回填条数相等");
    expect_true(stats.leaf_cache_rate() > 0.0, "缓存应当命中过（同一局面会重复出现）");
}

TEST(engine_is_deterministic_for_same_config) {
    // 同一配置两次运行必须产出完全相同的分片。lane 调度顺序不影响写入内容，
    // 这是“数据可信”的基础性质；合成结果（不依赖浮点推理）让它可以被严格断言。
    TempDir dir("determinism");
    EngineConfig config;
    config.lanes = 4;
    config.simulations = 8;
    config.in_flight_per_lane = 2;
    config.cache_entries = 1 << 12;
    config.seed = 12345;

    std::vector<std::string> hashes_first;
    std::vector<std::string> hashes_second;
    for (int run = 0; run < 2; run++) {
        const std::string data_dir = dir.sub(run == 0 ? "a" : "b");
        BatchedEngine engine(config, data_dir, "determinism", "synthetic", 2000, 500);
        SyntheticDriver driver{engine};
        expect_true(driver.run_until_done(), "两次运行都要正常结束");
        engine.finish();
        const BatchManifest manifest = read_manifest(data_dir);
        std::vector<std::string> hashes;
        for (const ShardInfo& shard : manifest.shards) hashes.push_back(shard.sha256);
        if (run == 0) {
            hashes_first = hashes;
        } else {
            hashes_second = hashes;
        }
    }
    expect_true(!hashes_first.empty(), "至少产出一个分片");
    expect_true(hashes_first == hashes_second, "同配置两次运行的每个分片哈希都相同");
}

}  // namespace r2048::tests
