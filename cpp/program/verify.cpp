#include "program/verify.h"

#include <filesystem>

#include "core/fileio.h"
#include "core/hash.h"
#include "core/logging.h"
#include "dataio/sample.h"
#include "dataio/shard.h"
#include "program/generate.h"

namespace r2048 {

namespace {

std::vector<uint8_t> read_binary(const std::string& path) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) throw std::runtime_error("无法读取文件: " + path);
    std::vector<uint8_t> data;
    uint8_t buffer[1 << 16];
    size_t read = 0;
    while ((read = std::fread(buffer, 1, sizeof(buffer), file)) > 0) {
        data.insert(data.end(), buffer, buffer + read);
    }
    const bool failed = std::ferror(file) != 0;
    std::fclose(file);
    if (failed) throw std::runtime_error("读取文件失败: " + path);
    return data;
}

}  // namespace

VerifyReport verify_batch(const std::string& data_dir, const VerifyOptions& options) {
    VerifyReport report;
    const double start = monotonic_seconds();

    BatchManifest manifest;
    try {
        manifest = read_manifest(data_dir);
    } catch (const std::exception& error) {
        report.problems.push_back(std::string("清单读取失败: ") + error.what());
        report.seconds = monotonic_seconds() - start;
        return report;
    }
    report.samples = manifest.samples;
    report.episodes = manifest.episodes;
    report.shards = manifest.shards.size();
    report.state = batch_state_name(manifest.state);
    // 未跑完的批次是正常状态（被 pause/stop 叫停），此时它本来就短、也没有“应该等于目标条数”
    // 这回事——把“没跑完”报成“校验失败”只会让人学会忽略校验输出。
    const bool complete = manifest.state == BatchState::kComplete;
    if (manifest.shards.empty()) {
        if (complete) report.problems.push_back("已完成的批次里没有任何分片");
    }
    if (complete && manifest.meta.samples_target > 0 && manifest.samples != manifest.meta.samples_target) {
        report.problems.push_back("清单自称已完成，但样本数 " + std::to_string(manifest.samples) +
                                  " 不等于目标 " + std::to_string(manifest.meta.samples_target));
    }
    if (!complete && manifest.episodes == 0 && manifest.samples > 0) {
        report.problems.push_back("清单没有提交任何局，却有样本条数");
    }

    uint64_t total_samples = 0;
    for (size_t index = 0; index < manifest.shards.size(); index++) {
        const ShardInfo& shard = manifest.shards[index];
        const std::string path = data_dir + "/" + shard.path;
        const std::string tag = "分片 " + shard.path;

        if (!path_exists(path)) {
            report.problems.push_back(tag + " 不存在");
            continue;
        }
        const uint64_t bytes = file_size(path);
        if (bytes != shard.samples * kSampleBytes) {
            report.problems.push_back(tag + " 字节数与记录数不符：" + std::to_string(bytes) + " != " +
                                      std::to_string(shard.samples * kSampleBytes));
            continue;
        }
        // 除最后一片外，分片都必须攒够容量：分片偏小会让“分片边界只由样本流决定”失去意义。
        // 判据是 >= 而不是 ==：切分只在局边界上进行，所以分片通常会略微超过容量。
        if (index + 1 < manifest.shards.size() && shard.samples < manifest.meta.shard_samples) {
            report.problems.push_back(tag + " 未攒满容量（" + std::to_string(shard.samples) + " < " +
                                      std::to_string(manifest.meta.shard_samples) + "）");
        }

        const std::vector<uint8_t> data = read_binary(path);
        const std::string actual_hash = sha256_hex(data.data(), data.size());
        if (actual_hash != shard.sha256) {
            report.problems.push_back(tag + " 摘要不匹配：记录 " + shard.sha256 + "，实际 " + actual_hash);
        }

        // 记录内容：局号必须单调不减，且分片首尾局号与清单一致。
        uint32_t previous_episode = 0;
        bool first_record = true;
        for (uint64_t offset = 0; offset + kSampleBytes <= data.size(); offset += kSampleBytes) {
            const Sample sample = decode_sample(data.data() + offset);
            if (!first_record && sample.episode < previous_episode) {
                report.problems.push_back(tag + " 内局号并非单调不减（记录序号 " +
                                          std::to_string(offset / kSampleBytes) + "）");
                break;
            }
            if (first_record) {
                if (sample.episode != shard.first_episode) {
                    report.problems.push_back(tag + " 首条记录的局号 " + std::to_string(sample.episode) +
                                              " 与清单 " + std::to_string(shard.first_episode) + " 不符");
                }
                first_record = false;
            }
            previous_episode = sample.episode;
        }
        if (!first_record && previous_episode != shard.last_episode) {
            report.problems.push_back(tag + " 末条记录的局号 " + std::to_string(previous_episode) +
                                      " 与清单 " + std::to_string(shard.last_episode) + " 不符");
        }
        total_samples += shard.samples;
    }

    if (total_samples != manifest.samples) {
        report.problems.push_back("分片样本总数 " + std::to_string(total_samples) + " 与清单 " +
                                  std::to_string(manifest.samples) + " 不符");
    }
    if (!manifest.shards.empty()) {
        const uint32_t last_episode = manifest.shards.back().last_episode;
        if (last_episode + 1 != manifest.episodes) {
            report.problems.push_back("末分片局号 " + std::to_string(last_episode) + " 与清单局数 " +
                                      std::to_string(manifest.episodes) + " 不一致");
        }
        if (manifest.shards.front().first_episode != 0) {
            report.problems.push_back("首发分片的起始局号不是 0");
        }
    }

    // 重放校验有两个前提，缺一个就不该硬做——做了只会给出错误的结论：
    //   * 切分规则必须与当前实现一致：规则变了，哈希当然不同，那不是“数据错了”；
    //   * 这一批必须已经跑完：未跑完的批次没有确定终点，而不限量的批次（目标 0）
    //     更是永远跑不到头，重放会一直生成下去。
    const bool replayable = manifest.shard_layout == kShardLayout && complete &&
                            manifest.meta.samples_target > 0;
    if (options.replay && !replayable) {
        report.replay_note = manifest.shard_layout != kShardLayout
                                 ? ("该批由旧的分片切分规则产出（shard_layout=" +
                                    std::to_string(manifest.shard_layout) + "），当前实现无法重放比对")
                                 : ("该批未跑完（state=" + report.state +
                                    "）或样本目标不限量，重放没有可比的终点");
    }

    if (options.replay && replayable && report.problems.empty()) {
        const bool auto_dir = options.replay_dir.empty();
        const std::string replay_dir = auto_dir ? data_dir + "_replay" : options.replay_dir;
        std::filesystem::remove_all(replay_dir);

        GenerateConfig config;
        config.data_dir = replay_dir;
        config.batch_id = manifest.meta.batch_id;
        config.policy = manifest.meta.policy;
        config.simulations = manifest.meta.simulations;
        config.temperature_moves = manifest.meta.temperature_moves;
        config.samples_target = manifest.meta.samples_target;
        config.seed = manifest.meta.seed;
        config.shard_samples = manifest.meta.shard_samples;
        config.workers = options.replay_workers;
        config.model_id = manifest.meta.model_id;

        BatchManifest replayed;
        try {
            run_generate(config);
            replayed = read_manifest(replay_dir);
        } catch (const std::exception& error) {
            report.problems.push_back(std::string("重放失败: ") + error.what());
        }

        if (replayed.shards.size() != manifest.shards.size()) {
            report.problems.push_back("重放的分片数 " + std::to_string(replayed.shards.size()) +
                                      " 与原始 " + std::to_string(manifest.shards.size()) + " 不符");
        } else {
            for (size_t index = 0; index < manifest.shards.size(); index++) {
                if (replayed.shards[index].sha256 != manifest.shards[index].sha256) {
                    report.problems.push_back("重放的第 " + std::to_string(index) + " 个分片哈希不一致" +
                                              "（线程数 " + std::to_string(options.replay_workers) +
                                              "，原始 " + std::to_string(manifest.meta.workers) + "）");
                }
                if (replayed.shards[index].samples != manifest.shards[index].samples) {
                    report.problems.push_back("重放的第 " + std::to_string(index) + " 个分片条数不一致");
                }
            }
        }
        if (replayed.samples != manifest.samples || replayed.episodes != manifest.episodes) {
            report.problems.push_back("重放的样本数或局数与原始清单不符");
        }
        report.replay_checked = true;
        if (auto_dir) std::filesystem::remove_all(replay_dir);
    }

    report.ok = report.problems.empty();
    report.seconds = monotonic_seconds() - start;
    return report;
}

}  // namespace r2048
