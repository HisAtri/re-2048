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
    std::vector<BatchedEngine::Advance> batches;  // 保留每轮的返回（含已结束的局）

    bool run_until_done(uint64_t max_rounds = 200000) {
        std::vector<uint64_t> ids;
        std::vector<float> values;
        std::vector<float> logits;
        while (rounds < max_rounds) {
            rounds++;
            BatchedEngine::Advance advance = engine.advance(4096);
            const size_t count = advance.request_ids.size();
            if (!advance.episodes.empty()) batches.push_back(advance);
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


TEST(engine_greedy_modes_skip_search_and_report_outcomes) {
    // 评测模式：关闭样本收集、按局数停止，并且每局结束都回报结果。
    // 合成结果（价值恒 0、logits 恒 0）让三种模式的行为可以精确推断。
    struct Run {
        uint64_t episodes = 0;
        uint64_t simulations = 0;
        uint64_t steps = 0;
        uint64_t total_score = 0;
        uint64_t requests = 0;
    };
    const auto run_mode = [](DecisionMode mode) {
        EngineConfig config;
        config.lanes = 4;
        config.simulations = 64;  // 贪心模式应当忽略它
        config.in_flight_per_lane = 4;
        config.cache_entries = 1 << 12;
        config.seed = 20261006;
        config.decision_mode = mode;
        config.collect_samples = false;
        config.episodes_target = 6;
        // 评测模式不写任何数据，因此不需要输出目录。
        BatchedEngine engine(config, "", "", "synthetic", 0, 0);
        SyntheticDriver driver{engine};
        const bool done = driver.run_until_done();
        expect_true(done, "评测运行应当在有限轮次内跑满目标局数");
        Run run;
        run.requests = driver.requests;
        for (const BatchedEngine::Advance& batch : driver.batches) {
            for (const EpisodeResult& outcome : batch.episodes) {
                run.episodes++;
                run.steps += outcome.steps;
                run.total_score += outcome.score;
                expect_true(outcome.steps > 0, "每局至少走了一步");
                expect_true(outcome.max_exponent >= 1, "开局至少有一个块");
            }
        }
        const EngineStats stats = engine.stats();
        run.simulations = stats.simulations;
        expect_equal_u64(stats.samples, 0, "评测模式不写样本");
        return run;
    };

    const Run value_run = run_mode(kDecisionValueGreedy);
    expect_equal_u64(value_run.episodes, 6, "恰好跑满目标局数");
    // 贪心模式每步只求值一次根局面，因此模拟次数等于总步数——这就是“没有搜索”的定量表达。
    expect_equal_u64(value_run.simulations, value_run.steps, "贪心模式每步仅一次根求值");
    expect_true(value_run.requests >= value_run.steps, "每步至少一条请求（策略 + afterstate 价值）");

    const Run policy_run = run_mode(kDecisionPolicyGreedy);
    expect_equal_u64(policy_run.episodes, 6, "策略贪心同样跑满目标局数");
    expect_equal_u64(policy_run.simulations, policy_run.steps, "策略贪心同样不做搜索");

    // 合成结果下两种贪心的行为差异是可推断的：价值恒 0 时“值贪心”等价于“最大化本次合并得分”，
    // 而 logits 恒 0 时“策略贪心”退化为按固定顺序取第一个合法方向（盲目）。
    expect_true(value_run.total_score > policy_run.total_score,
                "值贪心（取 r 最大）在合成结果下应当明显优于盲目的策略贪心");
}

TEST(engine_search_mode_is_unaffected_by_eval_switch) {
    // 搜索模式 + 关闭样本收集：仍然按局数停止，但不写数据、模拟次数远大于步数。
    EngineConfig config;
    config.lanes = 2;
    config.simulations = 8;
    config.in_flight_per_lane = 2;
    config.cache_entries = 1 << 12;
    config.seed = 99;
    config.collect_samples = false;
    config.episodes_target = 2;
    BatchedEngine engine(config, "", "", "synthetic", 0, 0);
    SyntheticDriver driver{engine};
    expect_true(driver.run_until_done(), "应当在有限轮次内跑满目标局数");

    uint64_t episodes = 0;
    uint64_t steps = 0;
    for (const BatchedEngine::Advance& batch : driver.batches) {
        for (const EpisodeResult& outcome : batch.episodes) {
            episodes++;
            steps += outcome.steps;
        }
    }
    expect_equal_u64(episodes, 2, "跑满两局");
    const EngineStats stats = engine.stats();
    expect_equal_u64(stats.samples, 0, "没有写入任何样本");
    expect_true(stats.simulations == steps * config.simulations,
                "搜索模式下模拟次数 = 步数 × 每步模拟次数");
}

}  // namespace r2048::tests
