#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "core/fileio.h"
#include "core/json.h"
#include "dataio/sample.h"
#include "game/rules.h"
#include "dataio/shard.h"
#include "program/episode.h"
#include "program/generate.h"
#include "program/verify.h"
#include "test_util.h"

namespace r2048::tests {

namespace {

// 每个用例用独立目录：ctest 可能并行跑多个套件，路径不能共用。
class TempDir {
public:
    explicit TempDir(const std::string& name) {
        path_ = (std::filesystem::temp_directory_path() / ("r2048-test-" + name)).string();
        std::filesystem::remove_all(path_);
        ensure_dir(path_);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    std::string path() const { return path_; }
    std::string sub(const std::string& name) const { return path_ + "/" + name; }

private:
    std::string path_;
};

GenerateConfig small_config(const std::string& out_dir, const std::string& policy, uint32_t workers) {
    GenerateConfig config;
    config.data_dir = out_dir;
    config.batch_id = "test-batch";
    config.policy = policy;
    config.simulations = policy == "mcts" ? 8 : 0;
    config.temperature_moves = policy == "mcts" ? 2 : 0;
    config.samples_target = 5000;
    config.seed = 20261006;
    config.shard_samples = 2000;
    config.workers = workers;
    return config;
}

std::vector<std::string> shard_hashes(const std::string& data_dir) {
    const BatchManifest manifest = read_manifest(data_dir);
    std::vector<std::string> hashes;
    for (const ShardInfo& shard : manifest.shards) hashes.push_back(shard.sha256);
    return hashes;
}

}  // namespace

TEST(dataio_sample_roundtrip) {
    Sample sample;
    for (int i = 0; i < 16; i++) {
        sample.decision.tiles[i] = static_cast<uint8_t>(i);
        sample.afterstate.tiles[i] = static_cast<uint8_t>(15 - i);
    }
    sample.action = 2;
    sample.legal_mask = 0b1101;
    sample.step = 1234;
    sample.episode = 99999;
    sample.simulations = 64;
    sample.move_score = 4096;
    sample.mc_return = -1234567890123LL;
    for (int action = 0; action < kNumActions; action++) sample.visits[action] = static_cast<uint16_t>(action + 1);
    sample.candidates.count = 3;
    for (int action = 0; action < kNumActions; action++) {
        for (int cell = 0; cell < kBoardCells; cell++) {
            sample.candidates.afterstate[action].tiles[cell] = static_cast<uint8_t>(action * 10 + cell);
        }
        sample.candidates.value[action] = action * 1000 - 500;   // 含负值
        sample.candidates.reward[action] = static_cast<uint32_t>(action * 7);
        sample.candidates.visits[action] = static_cast<uint16_t>(action * 3);
    }

    uint8_t encoded[kSampleBytes];
    encode_sample(sample, encoded);
    const Sample decoded = decode_sample(encoded);

    expect_true(decoded.decision == sample.decision, "决策局面往返一致");
    expect_true(decoded.afterstate == sample.afterstate, "afterstate 往返一致");
    expect_equal_u64(decoded.action, sample.action, "动作往返一致");
    expect_equal_u64(decoded.legal_mask, sample.legal_mask, "合法掩码往返一致");
    expect_equal_u64(decoded.step, sample.step, "步号往返一致");
    expect_equal_u64(decoded.episode, sample.episode, "局号往返一致");
    expect_equal_u64(decoded.simulations, sample.simulations, "模拟次数往返一致");
    expect_equal_u64(decoded.move_score, sample.move_score, "合并得分往返一致");
    expect_true(decoded.mc_return == sample.mc_return, "回报标签往返一致（含负数）");
    for (int action = 0; action < kNumActions; action++) {
        expect_equal_u64(decoded.visits[action], sample.visits[action], "访问数往返一致");
    }

    expect_equal_u64(decoded.candidates.count, 3, "候选数量往返一致");
    for (int action = 0; action < kNumActions; action++) {
        expect_true(decoded.candidates.afterstate[action] == sample.candidates.afterstate[action],
                    "候选 afterstate 往返一致");
        expect_true(decoded.candidates.value[action] == sample.candidates.value[action],
                    "候选价值目标往返一致（含负值）");
        expect_equal_u64(decoded.candidates.reward[action], sample.candidates.reward[action],
                         "候选立即得分往返一致");
        expect_equal_u64(decoded.candidates.visits[action], sample.candidates.visits[action],
                         "候选访问数往返一致");
    }

    // 保留字节必须写 0：否则未来加字段时无法区分“旧数据未使用”与“旧的垃圾值”。
    for (size_t offset = 72; offset < 80; offset++) expect_equal_u64(encoded[offset], 0, "保留字节为 0");
    for (size_t offset = 185; offset < kSampleBytes; offset++) {
        expect_equal_u64(encoded[offset], 0, "v2 保留字节为 0");
    }
}

TEST(dataio_recorder_labels) {
    // 回报标签口径：第 t 步的标签 = t 步之后还能拿到的分数，不含 t 步本身；最后一步为 0。
    EpisodeRecorder recorder(7, 64);
    const uint16_t no_visits[kNumActions] = {0, 0, 0, 0};
    const uint32_t scores[4] = {4, 0, 16, 8};
    for (int step = 0; step < 4; step++) {
        Board decision = Board::empty();
        Board afterstate = Board::empty();
        decision.tiles[0] = static_cast<uint8_t>(step + 1);
        afterstate.tiles[1] = static_cast<uint8_t>(step + 1);
        recorder.add(decision, afterstate, step % kNumActions, 0b1111, scores[step], no_visits);
    }
    const std::vector<Sample> samples = recorder.finish();
    expect_equal_u64(samples.size(), 4, "四条记录");
    expect_equal_u64(static_cast<uint64_t>(samples[3].mc_return), 0, "最后一步的标签为 0");
    expect_equal_u64(static_cast<uint64_t>(samples[2].mc_return), 8, "倒数第二步只含最后一步得分");
    expect_equal_u64(static_cast<uint64_t>(samples[1].mc_return), 24, "第三步含后两步得分");
    expect_equal_u64(static_cast<uint64_t>(samples[0].mc_return), 24, "首步不含本步得分（24 而非 28）");
    for (size_t index = 0; index < samples.size(); index++) {
        expect_equal_u64(samples[index].episode, 7, "局号写入每条记录");
        expect_equal_u64(samples[index].step, index, "步号按顺序写入");
    }
}

TEST(dataio_manifest_json_roundtrip) {
    BatchManifest manifest;
    manifest.meta.batch_id = "mcts-20261006-01";
    manifest.meta.policy = "mcts";
    manifest.meta.simulations = 64;
    manifest.meta.temperature_moves = 30;
    manifest.meta.seed = 20261006;
    manifest.meta.shard_samples = 100000;
    manifest.meta.workers = 16;
    manifest.meta.samples_target = 200000;
    manifest.meta.model_id = "s3p3v1-c64-0001";
    manifest.meta.params = {{"policy", "mcts"}, {"samples", "200000"}};
    manifest.samples = 200000;
    manifest.episodes = 1697;
    manifest.shards = {{"shards/shard-000000.bin", 100000, 0, 812, "aa"}, {"shards/shard-000001.bin", 100000, 812, 1696, "bb"}};

    const std::string text = manifest_to_json(manifest).dump(2);
    const BatchManifest parsed = manifest_from_json(parse_json(text));

    expect_true(parsed.meta.batch_id == manifest.meta.batch_id, "批次号往返一致");
    expect_equal_u64(parsed.meta.simulations, 64, "模拟次数往返一致");
    expect_equal_u64(parsed.meta.temperature_moves, 30, "温度步数往返一致");
    expect_equal_u64(parsed.meta.seed, 20261006, "种子往返一致");
    expect_equal_u64(parsed.meta.shard_samples, 100000, "分片容量往返一致");
    expect_equal_u64(parsed.samples, 200000, "样本总数往返一致");
    expect_equal_u64(parsed.episodes, 1697, "局数往返一致");
    expect_equal_u64(parsed.shards.size(), 2, "分片数往返一致");
    expect_contains(parsed.shards[1].sha256, "bb", "分片摘要往返一致");
    expect_equal_u64(parsed.shards[1].first_episode, 812, "分片起始局号往返一致");
    expect_equal_u64(parsed.meta.params.size(), 2, "原始参数往返一致");

    // 版本不匹配必须报错：用错语义解释旧数据比直接失败危险得多。
    const std::string broken = text.substr(0, text.find("\"schema_version\"")) + "\"schema_version\": 99, " +
                               text.substr(text.find("\"rules_version\""));
    expect_contains(catch_message([&] { (void)manifest_from_json(parse_json(broken)); }), "版本不匹配",
                    "记录格式版本不匹配时报错");
}

TEST(dataio_generate_deterministic_across_workers) {
    // 核心不变式：数据文件逐字节只由 (种子, 参数, 目标条数) 决定，与 worker 数量无关。
    TempDir dir("determinism");
    const std::string single = dir.sub("workers1");
    const std::string many = dir.sub("workers8");
    run_generate(small_config(single, "random", 1));
    run_generate(small_config(many, "random", 8));

    const BatchManifest manifest_single = read_manifest(single);
    const BatchManifest manifest_many = read_manifest(many);
    expect_equal_u64(manifest_single.samples, 5000, "样本数恰好等于目标条数");
    expect_equal_u64(manifest_many.samples, 5000, "并行生成同样恰好等于目标条数");
    expect_equal_u64(manifest_single.episodes, manifest_many.episodes, "局数与线程数无关");
    expect_equal_u64(manifest_single.shards.size(), manifest_many.shards.size(), "分片数与线程数无关");
    expect_true(shard_hashes(single) == shard_hashes(many), "不同线程数产出的分片逐字节相同");

    // 换种子必须换数据，否则说明种子没有真正参与随机流。
    const std::string other = dir.sub("other-seed");
    GenerateConfig config = small_config(other, "random", 4);
    config.seed = 999;
    run_generate(config);
    expect_true(shard_hashes(other) != shard_hashes(single), "不同种子产出不同数据");
}

TEST(dataio_verify_detects_corruption) {
    TempDir dir("verify");
    const std::string data_dir = dir.sub("batch");
    run_generate(small_config(data_dir, "random", 4));

    const VerifyReport ok_report = verify_batch(data_dir, VerifyOptions{});
    expect_true(ok_report.ok, "刚生成的批次应通过校验");

    // 重放校验：用不同的线程数复现整批数据，哈希必须一致。
    VerifyOptions replay;
    replay.replay_workers = 3;
    const VerifyReport replay_report = verify_batch(data_dir, replay);
    expect_true(replay_report.ok, "重放校验应通过");
    expect_true(replay_report.replay_checked, "确实执行了重放");

    // 篡改分片内容：摘要校验必须发现。
    const BatchManifest manifest = read_manifest(data_dir);
    const std::string shard_path = data_dir + "/" + manifest.shards.front().path;
    std::string bytes = read_text_file(shard_path);
    bytes[100] = static_cast<char>(bytes[100] ^ 0x5A);
    write_text_atomic(shard_path, bytes);
    const VerifyReport tampered = verify_batch(data_dir, VerifyOptions{false, 1, ""});
    expect_true(!tampered.ok, "被篡改的分片必须校验失败");
    expect_contains(join_text(tampered.problems), "摘要不匹配", "失败原因指向摘要");

    // 篡改清单条数：结构交叉校验必须发现。
    run_generate(small_config(dir.sub("batch2"), "random", 4));
    const std::string manifest_path = dir.sub("batch2") + "/manifest.json";
    std::string text = read_text_file(manifest_path);
    const std::string from = "\"samples\": 5000";
    const size_t found = text.find(from);
    expect_true(found != std::string::npos, "清单里能找到样本数字段");
    text.replace(found, from.size(), "\"samples\": 4999");
    write_text_atomic(manifest_path, text);
    const VerifyReport broken = verify_batch(dir.sub("batch2"), VerifyOptions{false, 1, ""});
    expect_true(!broken.ok, "清单条数被改动后必须校验失败");
}

TEST(dataio_generate_refuses_to_overwrite) {
    TempDir dir("overwrite");
    const std::string data_dir = dir.sub("batch");
    run_generate(small_config(data_dir, "random", 2));
    // 同一个目录再跑一次必须直接失败：跑谱批按“整批要么完整、要么重来”管理。
    expect_contains(catch_message([&] { run_generate(small_config(data_dir, "random", 2)); }), "拒绝覆盖",
                    "拒绝覆盖已有数据批");
}

TEST(dataio_candidate_targets_arithmetic) {
    // 候选目标的定义：A(u_a) = Q̄(a) − r(s,a)。这里用给定的 Q̄ 直接核对算术、
    // afterstate 与立即得分是否都由规则正确补齐。
    Board decision = Board::empty();
    decision.at(0, 0) = 1;
    decision.at(0, 1) = 1;   // 可以合并
    decision.at(1, 3) = 2;
    const uint8_t mask = legal_mask(decision);

    int32_t q_mean[kNumActions] = {0, 0, 0, 0};
    uint16_t visits[kNumActions] = {0, 0, 0, 0};
    for (int action = 0; action < kNumActions; action++) {
        if ((mask & (1u << action)) == 0) continue;
        q_mean[action] = 5000 + 100 * action;   // 故意取与立即得分不同的量级
        visits[action] = static_cast<uint16_t>(10 + action);
    }
    const CandidateTargets targets = build_candidate_targets(decision, mask, q_mean, visits);

    int legal = 0;
    for (int action = 0; action < kNumActions; action++) {
        if ((mask & (1u << action)) == 0) {
            expect_equal_u64(targets.value[action], 0, "非法方向的候选槽保持为空");
            continue;
        }
        legal++;
        const MoveOutcome outcome = apply_move(decision, action);
        expect_true(targets.afterstate[action] == outcome.board, "候选 afterstate 由规则补齐");
        expect_equal_u64(targets.reward[action], outcome.score, "候选立即得分由规则补齐");
        expect_equal_u64(targets.visits[action], visits[action], "候选访问数原样保留");
        expect_equal_u64(static_cast<uint64_t>(targets.value[action]),
                         static_cast<uint64_t>(q_mean[action] - static_cast<int32_t>(outcome.score)),
                         "候选价值目标 = Q̄(a) − r(s,a)");
    }
    expect_equal_u64(targets.count, static_cast<uint64_t>(legal), "候选数量等于合法方向数");
}

}  // namespace r2048::tests
