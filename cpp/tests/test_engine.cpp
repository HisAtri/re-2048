#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/fileio.h"
#include "dataio/sample.h"
#include "dataio/shard.h"
#include "game/rules.h"
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

// 驱动一次"会话分析"：begin_analysis 之后反复 advance/submit，直到引擎说 done。
// 合成结果（价值恒 0、先验均匀）让报告里的数可以被精确断言。
bool run_analysis(BatchedEngine& engine, uint32_t budget) {
    engine.begin_analysis(budget);
    for (uint64_t round = 0; round < 200000; round++) {
        const BatchedEngine::Advance advance = engine.advance(4096);
        const size_t count = advance.request_ids.size();
        if (count > 0) {
            std::vector<float> values(count, 0.0f);
            std::vector<float> logits(count * kNumActions, 0.0f);
            engine.submit(advance.request_ids.data(), values.data(), logits.data(), count);
            continue;
        }
        return advance.done;
    }
    return false;
}

std::vector<uint8_t> read_binary_for_test(const std::string& path) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) throw std::runtime_error("无法打开 " + path);
    std::vector<uint8_t> data;
    uint8_t buffer[1 << 16];
    size_t read = 0;
    while ((read = std::fread(buffer, 1, sizeof(buffer), file)) > 0) {
        data.insert(data.end(), buffer, buffer + read);
    }
    std::fclose(file);
    return data;
}

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

    const FinishResult finished = engine.finish();
    const BatchManifest manifest = read_manifest(dir.sub("batch"));
    expect_equal_u64(manifest.samples, 3000, "样本数恰好等于目标");
    expect_true(manifest.meta.policy == "nn", "策略标记为 nn");
    expect_true(manifest.meta.model_id == "synthetic", "来源模型记录在清单里");
    // 局数与"最高局号 + 1"不再相等：多个 lane 同时开局、只有一部分走完，所以局号有洞
    // （发号器按块预留的设计使然，见 ADR 0029）。不变量是"局数 = 各分片区间之和"，
    // 并且整批最小的局号是 0。
    uint64_t covered = 0;
    uint32_t lowest = UINT32_MAX;
    uint32_t highest = 0;
    for (const ShardInfo& shard : manifest.shards) {
        covered += shard.episodes;
        lowest = std::min(lowest, shard.first_episode);
        highest = std::max(highest, shard.last_episode);
    }
    expect_equal_u64(manifest.episodes, covered, "局数 = 各分片记录的局数之和");
    expect_equal_u64(lowest, 0, "整批最小的局号必须是 0");
    expect_true(manifest.next_episode > highest, "清单的 next_episode 必须是高水位");
    expect_contains(finished.manifest_path, "manifest.json", "返回 manifest 路径");
    expect_true(finished.state == BatchState::kComplete, "跑满目标时状态为 complete");

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

TEST(engine_abandon_requeues_dropped_batch) {
    // 一次被丢掉的前向必须能被补救：Ctrl-C 恰好落在前向里、Python 侧回调抛异常、
    // 显存不足，都会让"已发出但未回填"的请求留在引擎里。若它们不被重发，这些 lane
    // 就永远等不到答案——调用方看到的是"没有请求、也没有结束"，只能卡死或放弃整批数据。
    //
    // 补救现在是**显式**的（`abandon()`）：管线重叠允许"手上有未回填的批次、同时继续
    // advance"，所以引擎不能再自动补发——那会把正在 GPU 上算的批次当成丢失的重复发一遍
    // （重复 id 会让 submit 直接抛错）。见 ADR 0027。
    TempDir dir("abandon");
    EngineConfig config;
    config.lanes = 4;
    config.simulations = 8;
    config.in_flight_per_lane = 2;
    config.cache_entries = 1 << 12;
    config.seed = 4242;

    BatchedEngine engine(config, dir.sub("batch"), "abandon-test", "synthetic", 1500, 500);
    uint64_t rounds = 0;
    bool done = false;
    while (rounds < 200000 && !done) {
        rounds++;
        BatchedEngine::Advance advance = engine.advance(4096);
        if (rounds == 3) {
            // 第 3 轮模拟"这一批被丢掉"：既不回填，也不报错；调用方随后声明放弃它。
            expect_true(!advance.request_ids.empty(), "被丢掉的那一轮确实带走了请求");
            engine.abandon();
            continue;
        }
        if (!advance.request_ids.empty()) {
            std::vector<float> values(advance.request_ids.size(), 0.0f);
            std::vector<float> logits(advance.request_ids.size() * kNumActions, 0.0f);
            engine.submit(advance.request_ids.data(), values.data(), logits.data(),
                          advance.request_ids.size());
        } else if (advance.done) {
            done = true;
        }
    }
    expect_true(done, "声明放弃一批之后引擎仍然能跑完（而不是永远没有请求可发）");

    const EngineStats stats = engine.stats();
    expect_equal_u64(stats.requests_issued, stats.requests_submitted,
                     "最终每个请求都恰好回填一次（补发不会让请求数重复计）");
    engine.finish();
    const BatchManifest manifest = read_manifest(dir.sub("batch"));
    expect_equal_u64(manifest.samples, 1500, "补发之后样本数仍然恰好等于目标");
}

TEST(engine_advance_waits_when_budget_is_full) {
    // **边界**：`in_flight_per_lane` 就是引擎的管线深度。一次 advance 会把每条 lane 的在途
    // 预算用满，所以"手上还有未回填的批次时继续 advance"拿不到新请求——不是 bug，而是
    // "在途模拟互相看不见对方"这条搜索语义的直接后果（ADR 0017）。
    //
    // 这条测试钉住这个事实，有两个用处：
    //  1. 它解释了"为什么 GPU 会等 CPU"：管线深度是搜索参数，不能为了喂饱 GPU 随便加深
    //     （加深会改变数据，实测 lanes/in_flight 单独就能让评分位移 +20.1 分）；
    //  2. 谁将来要动并行化（把 lane 分组跑线程），这条测试会立刻告诉他改了什么。
    TempDir dir("budget");
    EngineConfig config;
    config.lanes = 4;
    config.simulations = 8;
    config.in_flight_per_lane = 4;
    config.cache_entries = 1 << 12;
    config.seed = 99;

    BatchedEngine engine(config, dir.sub("batch"), "budget-test", "synthetic", 3000, 1000);
    const BatchedEngine::Advance first = engine.advance(4096);
    expect_true(!first.request_ids.empty(), "第一轮应当把在途预算用满");
    const BatchedEngine::Advance second = engine.advance(4096);
    expect_true(second.request_ids.empty(),
                "预算用满后继续 advance 必须返回空批（要更多请求就得先回填或加深 in_flight）");
    expect_true(!second.done, "空批不等于收工");

    // 未回填的那一批**不能被重复发出**：id 不允许在后续批次里再出现。
    std::unordered_set<uint64_t> first_ids(first.request_ids.begin(), first.request_ids.end());
    for (const uint64_t id : second.request_ids) {
        expect_true(first_ids.count(id) == 0, "未回填的批次不该被重复发出");
    }

    // 回填之后立刻又能推进：这条保证"空批"不会变成死锁。
    std::vector<float> values(first.request_ids.size(), 0.0f);
    std::vector<float> logits(first.request_ids.size() * kNumActions, 0.0f);
    engine.submit(first.request_ids.data(), values.data(), logits.data(), first.request_ids.size());
    expect_true(!engine.advance(4096).request_ids.empty(), "回填之后必须能继续推进");

    engine.abandon();  // 收尾：把在途的清掉，避免运行结束时留着未回填的请求
    engine.finish();
}

TEST(engine_two_engines_share_one_sink) {
    // 多线程搜索的产出侧契约：K 个引擎（各持一份叶子缓存）**共享一个写入侧**，于是它们
    // 产出的必须是**同一批**数据——一个批次号、一份清单、一套连续局号，样本数恰好等于目标。
    // 这条不成立的话，多线程搜索就会把一批数据拆成 K 份（用户明确否掉的那种"很乱"）。
    TempDir dir("shared_sink");
    EngineConfig config;
    config.lanes = 4;        // 每个引擎 4 条 lane
    config.lanes_total = 8;  // 整批 8 条（清单与协议参数记这个）
    config.simulations = 8;
    config.in_flight_per_lane = 2;
    config.search_threads = 2;  // Python 侧的搜索线程数，进协议参数
    config.cache_entries = 1 << 12;
    config.seed = 4242;

    BatchedEngine first(config, dir.sub("batch"), "shared", "synthetic", 3000, 1000);
    const std::shared_ptr<SampleSink> sink = first.sink();
    expect_true(sink != nullptr, "第一个引擎应当自建写入侧");
    BatchedEngine second(config, dir.sub("batch"), "shared", "synthetic", 3000, 1000, sink);
    expect_true(second.sink() == sink, "第二个引擎必须共用同一个写入侧");

    BatchedEngine* engines[2] = {&first, &second};
    bool done[2] = {false, false};
    for (uint64_t rounds = 0; rounds < 200000 && !(done[0] && done[1]); rounds++) {
        for (int i = 0; i < 2; i++) {
            if (done[i]) continue;
            const BatchedEngine::Advance advance = engines[i]->advance(4096);
            if (!advance.request_ids.empty()) {
                std::vector<float> values(advance.request_ids.size(), 0.0f);
                std::vector<float> logits(advance.request_ids.size() * kNumActions, 0.0f);
                engines[i]->submit(advance.request_ids.data(), values.data(), logits.data(),
                                   advance.request_ids.size());
            } else if (advance.done) {
                done[i] = true;
            }
        }
    }
    expect_true(done[0] && done[1], "两个引擎都要跑到收尾");
    expect_equal_u64(sink->written_samples(), 3000, "共享写入侧的样本数必须恰好等于目标");
    expect_true(sink->written_episodes() > 0, "必须真的产出局");

    first.finish();  // 幂等：第二次收尾不会重复写
    const BatchManifest manifest = read_manifest(dir.sub("batch"));
    expect_equal_u64(manifest.samples, 3000, "清单里的样本数等于目标");
    expect_true(manifest.episodes > 0, "清单里有局");
    // 清单记的是**整批**的并发数与协议参数，而不是某一个引擎的。
    expect_equal_u64(manifest.meta.workers, 8, "workers 记整批的 lane 数（8）而不是单个引擎的（4）");
    std::string lanes_param, threads_param;
    for (const auto& entry : manifest.meta.params) {
        if (entry.first == "lanes") lanes_param = entry.second;
        if (entry.first == "search_threads") threads_param = entry.second;
    }
    expect_true(lanes_param == "8", "协议参数 lanes 也记整批的 8");
    expect_true(threads_param == "2", "协议参数 search_threads 记搜索线程数");
    expect_true(sink->stop_requested(), "收尾后写入侧必须处于停车状态");
}

TEST(engine_records_search_threads_and_conserves_requests) {
    // 引擎**内部不再并行**推进 lane（试过，实测负收益，见 ADR 0027）：并行走 K 个各自持有
    // 叶子缓存的引擎 + 一个合并推理线程（ADR 0028）。这里守的是与之配套的两件事：
    // 请求数守恒（一条都不能丢、不能重复回填）与清单如实记下搜索线程数（它改变调度，
    // 所以是协议参数）。
    TempDir dir("threads");
    EngineConfig config;
    config.lanes = 16;
    config.simulations = 8;
    config.in_flight_per_lane = 4;
    config.search_threads = 4;
    config.cache_entries = 1 << 12;
    config.seed = 31337;

    BatchedEngine engine(config, dir.sub("batch"), "threads-test", "synthetic", 4000, 1000);
    uint64_t rounds = 0;
    bool done = false;
    while (rounds < 200000 && !done) {
        rounds++;
        const BatchedEngine::Advance advance = engine.advance(4096);
        if (!advance.request_ids.empty()) {
            std::vector<float> values(advance.request_ids.size(), 0.0f);
            std::vector<float> logits(advance.request_ids.size() * kNumActions, 0.0f);
            engine.submit(advance.request_ids.data(), values.data(), logits.data(),
                          advance.request_ids.size());
        } else if (advance.done) {
            done = true;
        }
    }
    expect_true(done, "多线程推进也必须能跑到收尾");
    const EngineStats stats = engine.stats();
    expect_equal_u64(stats.requests_issued, stats.requests_submitted, "请求数必须守恒（线程不丢不重）");
    engine.finish();
    const BatchManifest manifest = read_manifest(dir.sub("batch"));
    expect_equal_u64(manifest.samples, 4000, "样本数仍然恰好等于目标");
    expect_true(manifest.episodes > 0, "必须真的产出局（局号连续性由写入侧的检查保证）");
    // 协议参数里必须记下线程数：它改变调度，因此和 lanes / in_flight 一样进清单。
    std::string recorded;
    for (const auto& entry : manifest.meta.params) {
        if (entry.first == "search_threads") recorded = entry.second;
    }
    expect_true(!recorded.empty(), "清单里必须记录 search_threads（它改变调度，属协议参数）");
    expect_true(recorded == "4", "记录的线程数必须与实际一致");
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
    expect_true(stats.simulations == steps * config.simulations, "搜索模式下模拟次数 = 步数 × 每步模拟次数");
}

TEST(engine_records_candidate_targets) {
    // 候选级监督的口径：为每个合法方向都写一条 A(u_a) = Q̄(a) − r(s,a)。
    // 合成网络的价值恒为 0，因此 Q̄(a) = r(s,a)，候选价值目标必须恒为 0——
    // 这条恒等式同时验证了“候选块被写入”“Q̄ 来自正确的边”“减的是这条边的立即得分”。
    TempDir dir("candidates");
    EngineConfig config;
    config.lanes = 2;
    config.simulations = 8;
    config.in_flight_per_lane = 2;
    config.cache_entries = 1 << 12;
    config.seed = 4242;

    BatchedEngine engine(config, dir.sub("batch"), "candidates", "synthetic", 400, 400);
    SyntheticDriver driver{engine};
    expect_true(driver.run_until_done(), "应当正常结束");
    engine.finish();

    const BatchManifest manifest = read_manifest(dir.sub("batch"));
    const std::vector<uint8_t> raw =
        read_binary_for_test(dir.sub("batch") + "/" + manifest.shards.front().path);
    int checked = 0;
    for (size_t offset = 0; offset + manifest.meta.record_bytes() <= raw.size();
         offset += manifest.meta.record_bytes()) {
        const Sample sample = decode_sample(raw.data() + offset, manifest.meta.tree_nodes);
        int legal = 0;
        for (int action = 0; action < kNumActions; action++) legal += (sample.legal_mask >> action) & 1;
        expect_equal_u64(sample.candidates.count, static_cast<uint64_t>(legal), "候选数量等于合法方向数");
        for (int action = 0; action < kNumActions; action++) {
            if (((sample.legal_mask >> action) & 1) == 0) continue;
            const MoveOutcome outcome = apply_move(sample.decision, action);
            expect_true(sample.candidates.afterstate[action] == outcome.board, "候选 afterstate 与规则一致");
            expect_equal_u64(sample.candidates.reward[action], outcome.score, "候选立即得分与规则一致");
            // 候选访问数与主记录的根访问数必须一致：两者独立写入，交叉核对能发现
            // “候选块按错的边填”这类错位。
            expect_equal_u64(sample.candidates.visits[action], sample.visits[action],
                             "候选访问数与根访问数一致");
            checked++;
        }
    }
    expect_true(checked > 0, "至少要检查到一条候选");
}

TEST(engine_tree_sampling_preserves_root_and_top_order) {
    // 两种采样只决定记录中的树内块；同一搜索协议下基础块必须逐字节一致。
    // top_visits 曾把较低访问数的节点插到高访问数节点前面，这里直接钉住槽位顺序。
    TempDir dir("tree-sampling");
    std::vector<std::vector<uint8_t>> batches;
    for (const uint8_t policy : {kTreeLabelTopVisits, kTreeLabelDepthStratified}) {
        EngineConfig config;
        config.lanes = 2;
        config.simulations = 32;
        // 单条在途模拟时边访问和与节点访问数只差固定的首次求值，便于从记录检验顺序。
        config.in_flight_per_lane = 1;
        config.cache_entries = 1 << 12;
        config.seed = 20261009;
        config.tree_nodes = 2;
        config.tree_label_policy = policy;
        config.tree_label_plays = 0;
        const std::string path = dir.sub(policy == kTreeLabelTopVisits ? "top" : "depth");
        BatchedEngine engine(config, path, "sampling", "synthetic", 400, 10000);
        SyntheticDriver driver{engine};
        expect_true(driver.run_until_done(), "树内采样测试应当正常完成");
        engine.finish();
        const BatchManifest manifest = read_manifest(path);
        batches.push_back(read_binary_for_test(path + "/" + manifest.shards.front().path));
    }

    expect_equal_u64(batches[0].size(), batches[1].size(), "采样策略不应改变记录条数");
    int checked = 0;
    for (size_t offset = 0; offset + sample_bytes(2) <= batches[0].size(); offset += sample_bytes(2)) {
        expect_true(std::memcmp(batches[0].data() + offset, batches[1].data() + offset, 185) == 0,
                    "采样策略不应改变根部基础记录");
        const Sample top = decode_sample(batches[0].data() + offset, 2);
        const Sample depth = decode_sample(batches[1].data() + offset, 2);
        if (top.tree_count < 2 || depth.tree_count < 2) continue;
        uint32_t visits[2] = {0, 0};
        for (int action = 0; action < kNumActions; action++) {
            visits[0] += top.tree[0].visits[action];
            visits[1] += top.tree[1].visits[action];
        }
        expect_true(visits[0] >= visits[1], "top_visits 的槽位必须按访问数降序");
        expect_true(depth.tree[0].depth < depth.tree[1].depth, "分层采样必须覆盖递增的两个决策深度");
        checked++;
    }
    expect_true(checked > 0, "至少有一条记录同时包含两个树内节点");
}

TEST(engine_rejects_tree_nodes_above_record_capacity) {
    EngineConfig config;
    config.tree_nodes = kTreeNodesMax + 1;
    config.collect_samples = false;
    expect_contains(
        catch_message([&] { BatchedEngine engine(config, "", "", "synthetic", 0, 0); }),
        "树内节点数超出上限", "选择数组的 K 必须在引擎启动前校验");
}

// ---- 会话驱动：任意局面 + 逐步走（ADR 0037）--------------------------------------
//
// 这一组测的是"从外部给定的局面起搜、一步一步走"这条路径。它此前完全不存在：整局驱动
// 每局都必须从 `opening_board` 派生，分析中间态也没有出口——而界面要回答的恰好是
// "这个盘面怎么样""AI 在这里会怎么走"。既有的六条整局驱动用例一条没改，就是这条路径
// 不影响它们的证据。

TEST(engine_session_analyzes_an_arbitrary_root) {
    EngineConfig config;
    config.lanes = 1;  // 会话只对一个盘面工作
    config.simulations = 16;
    config.in_flight_per_lane = 4;
    config.cache_entries = 1 << 12;
    config.seed = 20261008;
    config.temperature = 0.0f;       // 选招取最大访问数（不要开局采样），便于断言
    config.collect_samples = false;  // 会话不落样本
    BatchedEngine engine(config, "", "", "synthetic", 0, 0);

    Board root = Board::empty();
    root.tiles[0] = 1;
    root.tiles[5] = 2;
    root.tiles[10] = 3;
    engine.set_root(root);

    engine.begin_analysis(16);
    bool saw_root = false;
    uint64_t round = 0;
    for (; round < 200000; round++) {
        const BatchedEngine::Advance advance = engine.advance(4096);
        for (size_t i = 0; i < advance.request_ids.size(); i++) {
            Board requested;
            std::memcpy(requested.tiles, advance.boards.data() + i * kBoardCells, kBoardCells);
            if (requested == root) saw_root = true;
        }
        const size_t count = advance.request_ids.size();
        if (count > 0) {
            std::vector<float> values(count, 0.0f);
            std::vector<float> logits(count * kNumActions, 0.0f);
            engine.submit(advance.request_ids.data(), values.data(), logits.data(), count);
            continue;
        }
        if (advance.done) break;
        expect_true(false, "分析既没有请求也没有完成：引擎不可能自行前进");
        break;
    }
    expect_true(round < 200000, "分析应当在有限轮次内跑满预算");
    expect_true(saw_root, "给定的根局面必须真的被求值过（请求里要出现它）");

    const RootReport report = engine.report_root();
    expect_equal_u64(report.simulations, 16, "根上完成的模拟数等于预算");
    uint32_t total = 0;
    for (int action = 0; action < kNumActions; action++) total += report.visits[action];
    // 访问数之和比预算少 1：**恰好一条模拟用来求值根本身**（它停在根上，一条边都没走过）。
    // 这是"根节点还没被求值时只允许一条在途模拟"那条保护的结果（见 pump_lane_range），
    // 与整局驱动同源；写成 `total + 1 == simulations` 就是在钉"只丢一条、不多丢"。
    expect_equal_u64(static_cast<uint64_t>(total) + 1, report.simulations,
                     "各方向访问数之和 = 模拟数 − 1（恰好一条模拟用于求值根）");
    expect_equal_u64(report.legal_mask, legal_mask(root), "合法掩码来自给定的盘面");
    expect_true(report.best_action >= 0, "预算跑满后必须有一个被访问的方向");
}

TEST(engine_session_commit_step_spawns_and_reports_it) {
    // 玩家模式：按**给定**方向走一步。生成块由引擎自己生成并回报，所以"方块生成规则"
    // 仍然只有 C++ 一处实现——Python 侧不需要再实现一遍 90%/10% 的那些细节。
    EngineConfig config;
    config.lanes = 1;
    config.simulations = 8;
    config.in_flight_per_lane = 2;
    config.cache_entries = 1 << 12;
    config.seed = 777;
    config.collect_samples = false;
    BatchedEngine engine(config, "", "", "synthetic", 0, 0);

    Board root = Board::empty();
    root.tiles[1] = 1;  // b1
    root.tiles[3] = 1;  // d1：与 b1 同行，左移会合并
    root.tiles[8] = 2;  // a3
    engine.set_root(root);

    const MoveOutcome expected = apply_move(root, kLeft);
    expect_true(expected.valid, "这条用例的前提是左移合法");
    const StepResult step = engine.commit_step(kLeft);
    expect_equal_u64(static_cast<uint64_t>(step.action), static_cast<uint64_t>(kLeft),
                     "回报的方向就是给定方向");
    expect_equal_u64(step.score_delta, expected.score, "得分增量与规则一致");
    expect_equal_u64(step.step, 1, "会话的步号从 1 开始");
    expect_equal_u64(step.total_score, expected.score, "累计得分");
    expect_true(!step.terminal, "盘面还很大，不该终局");

    Board after;
    std::memcpy(after.tiles, step.board, kBoardCells);
    int differing = 0;
    for (int cell = 0; cell < kBoardCells; cell++) {
        if (after.tiles[cell] == expected.board.tiles[cell]) continue;
        differing++;
        expect_equal_u64(static_cast<uint64_t>(cell), static_cast<uint64_t>(step.spawn_index),
                         "与 afterstate 不同的那格就是回报的生成位置");
        expect_equal_u64(expected.board.tiles[cell], 0, "生成块落在空格上");
        expect_equal_u64(after.tiles[cell], step.spawn_exponent, "回报的生成块数值");
        expect_true(after.tiles[cell] == 1 || after.tiles[cell] == 2, "只会生成 2（指数 1）或 4（指数 2）");
    }
    expect_equal_u64(differing, 1, "一步之后与 afterstate 恰好只差生成的那一格");
}

TEST(engine_session_auto_choice_matches_the_report) {
    // 面板上显示的"引擎选了哪个方向"必须与引擎真的走的那一步一致——两者读的是同一份根
    // 统计。这条一旦不成立，界面就在骗人，而且骗得看不出来。
    EngineConfig config;
    config.lanes = 1;
    config.simulations = 16;
    config.in_flight_per_lane = 4;
    config.cache_entries = 1 << 12;
    config.seed = 4242;
    config.temperature = 0.0f;  // 否则开局步按访问分布采样，就不等于 argmax
    config.collect_samples = false;
    BatchedEngine engine(config, "", "", "synthetic", 0, 0);

    Board root = Board::empty();
    root.tiles[0] = 1;
    root.tiles[6] = 2;
    root.tiles[11] = 3;
    engine.set_root(root);
    expect_true(run_analysis(engine, 16), "分析应当跑满预算");

    const RootReport report = engine.report_root();
    expect_true(report.best_action >= 0, "分析之后必须有一个最佳方向");
    const StepResult step = engine.commit_step(-1);
    expect_equal_u64(static_cast<uint64_t>(step.action), static_cast<uint64_t>(report.best_action),
                     "引擎自己选的方向必须等于面板上显示的最佳方向");
    expect_true(!step.terminal, "这一步不该把盘面走死");
}

TEST(engine_session_terminal_step_does_not_open_a_new_game) {
    // 会话里终局就是收工：绝不能因为 `needs_new_episode` 让 pump 去开新局——那会把用户
    // 面前的残局悄悄换成一副随机开局，而看起来只是"棋盘忽然变了"。
    EngineConfig config;
    config.lanes = 1;
    config.simulations = 8;
    config.cache_entries = 1 << 10;
    config.collect_samples = false;
    BatchedEngine engine(config, "", "", "synthetic", 0, 0);

    // 相邻两格互不相同且没有空格：无合并、无移动 → 终局。
    const uint8_t pattern[kBoardCells] = {1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1};
    Board terminal = Board::empty();
    for (int cell = 0; cell < kBoardCells; cell++) terminal.tiles[cell] = pattern[cell];
    expect_equal_u64(legal_mask(terminal), 0, "这条用例的前提是该局面真的终局");

    engine.set_root(terminal);
    const StepResult step = engine.commit_step(-1);
    expect_true(step.terminal, "终局局面必须回报终局");
    expect_true(step.action == -1, "终局没有动作");
    expect_equal_u64(step.spawn_index, static_cast<uint64_t>(-1), "终局不生成块");

    // 终局之后再跑一轮：既不该有新请求（那会指向别的盘面），也不该冒出一局新对局。
    const BatchedEngine::Advance advance = engine.advance(4096);
    expect_true(advance.request_ids.empty(), "终局后不该再产生请求");
    expect_true(advance.episodes.empty(), "终局后不该冒出一局新的对局");

    // 重新摆一个盘面（set_root）之后又能工作——终局是可恢复的，不是卡死。
    Board fresh = Board::empty();
    fresh.tiles[5] = 1;
    engine.set_root(fresh);
    const StepResult resumed = engine.commit_step(kDown);
    expect_true(!resumed.terminal, "换成正常盘面后应当能继续走");
    expect_equal_u64(resumed.step, 1, "set_root 会把步号复位");
}

TEST(engine_session_rejects_batch_configurations) {
    // 会话是"一个盘面、一步一步走"，与并发跑局互斥。配错时必须立刻拒绝：静默接受会让
    // 调用方以为自己在逐步走，实际引擎却在并发跑局（而且多 lane 下"当前盘面"根本不是单数）。
    TempDir dir("session-guard");
    EngineConfig multi;
    multi.lanes = 2;
    multi.simulations = 8;
    multi.cache_entries = 1 << 10;
    multi.collect_samples = false;
    BatchedEngine wide(multi, "", "", "synthetic", 0, 0);
    expect_contains(catch_message([&] { wide.set_root(Board::empty()); }), "lanes==1",
                    "多 lane 配置必须被拒绝");
    expect_contains(catch_message([&] { wide.begin_analysis(4); }), "lanes==1", "同上");
    expect_contains(catch_message([&] { wide.report_root(); }), "lanes==1", "同上");
    expect_contains(catch_message([&] { wide.commit_step(-1); }), "lanes==1", "同上");

    EngineConfig sampling;
    sampling.lanes = 1;
    sampling.simulations = 8;
    sampling.cache_entries = 1 << 10;
    sampling.collect_samples = true;  // 错的：会话不落样本
    BatchedEngine collector(sampling, dir.sub("batch"), "guard", "synthetic", 100, 100);
    expect_contains(catch_message([&] { collector.set_root(Board::empty()); }), "不落样本",
                    "落样本的配置必须被拒绝");
}

}  // namespace r2048::tests
