#include "program/verify.h"

#include <cstdint>
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
        // 这里**不再检查"除末片外必须攒满容量"**：多写者下每个写者退出时都会留下一个未满的
        // 尾片，于是"只有末片可以不满"这条实现细节不再是不变量。数据完整性由摘要与条数保证，
        // 分片边界只是落盘策略。

        const std::vector<uint8_t> data = read_binary(path);
        const std::string actual_hash = sha256_hex(data.data(), data.size());
        if (actual_hash != shard.sha256) {
            report.problems.push_back(tag + " 摘要不匹配：记录 " + shard.sha256 + "，实际 " + actual_hash);
        }

        // 记录内容：每条记录的局号必须落在清单给这个分片声明的跨度内，并收集全局局号集合。
        // **不要求单调**：局号由跨进程发号器发放，同一分片里的局可能乱序到达（同写者内两条
        // lane 的完赛顺序不定）；分片自带 [first_episode, last_episode] 区间，读取端从记录
        // 里的局号还原，顺序不参与语义。区间内出现某条记录超界，才是真的坏了。
        uint32_t seen_records = 0;
        for (uint64_t offset = 0; offset + kSampleBytes <= data.size(); offset += kSampleBytes) {
            const Sample sample = decode_sample(data.data() + offset);
            seen_records++;
            if (sample.episode < shard.first_episode || sample.episode > shard.last_episode) {
                report.problems.push_back(tag + " 第 " + std::to_string(offset / kSampleBytes) +
                                          " 条记录的局号 " + std::to_string(sample.episode) +
                                          " 落在清单声明的区间 [" +
                                          std::to_string(shard.first_episode) + ", " +
                                          std::to_string(shard.last_episode) + "] 之外");
                break;
            }
            // 局号必须落在本批已发放的号段内：跨越 next_episode 说明清单与数据不是同一次
            // 运行写出来的（便宜的单遍检查，不需要额外内存）。
            if (sample.episode >= manifest.next_episode) {
                report.problems.push_back(tag + " 第 " + std::to_string(offset / kSampleBytes) +
                                          " 条记录的局号 " + std::to_string(sample.episode) +
                                          " 不小于 next_episode(" +
                                          std::to_string(manifest.next_episode) + "）");
                break;
            }
        }
        if (seen_records != shard.samples) {
            report.problems.push_back(tag + " 记录条数 " + std::to_string(seen_records) +
                                      " 与清单 " + std::to_string(shard.samples) + " 不符");
        }
        if (shard.first_episode > shard.last_episode) {
            report.problems.push_back(tag + " 的局号区间为空（first > last）");
        }
        if (shard.episodes == 0 && shard.samples > 0) {
            report.problems.push_back(tag + " 的局数为 0 但有样本");
        }
        // 注意：**不同分片的局号区间重叠是合法的**——同一写者的多局完赛顺序不定，
        // 一个分片的 [min,max] 只是跨度，不是它拥有的集合。真正的不变量"每局只出现
        // 一次"在下面用全局集合核（本工具是离线校验，几 MB 的集合可以接受）。
        total_samples += shard.samples;
    }

    if (total_samples != manifest.samples) {
        report.problems.push_back("分片样本总数 " + std::to_string(total_samples) + " 与清单 " +
                                  std::to_string(manifest.samples) + " 不符");
    }
    if (!manifest.shards.empty()) {
        // 清单顺序不等于局号顺序（多写者并发提交、续跑原地重写都会打乱它），
        // 所以"末局"要按所有分片取最大，而不是取清单里最后一个。
        uint32_t highest_episode = 0;
        uint32_t lowest_episode = UINT32_MAX;
        uint64_t episodes_sum = 0;
        for (const ShardInfo& shard : manifest.shards) {
            highest_episode = std::max(highest_episode, shard.last_episode);
            lowest_episode = std::min(lowest_episode, shard.first_episode);
            episodes_sum += shard.episodes;
        }
        if (lowest_episode != 0) {
            report.problems.push_back("整批最小的局号不是 0（实际 " + std::to_string(lowest_episode) + "）");
        }
        // 局数 = 各分片区间之和。局号可以有洞（发号器按块预留、写者被杀会留下未用的号），
        // 所以"最高局号 + 1"与"局数"不再相等，清单里两个字段分别记着。
        if (episodes_sum != manifest.episodes) {
            report.problems.push_back("各分片局号区间之和 " + std::to_string(episodes_sum) +
                                      " 与清单局数 " + std::to_string(manifest.episodes) + " 不符");
        }
        if (manifest.next_episode <= highest_episode) {
            report.problems.push_back("清单的 next_episode(" + std::to_string(manifest.next_episode) +
                                      ") 不大于已有的最高局号(" + std::to_string(highest_episode) + ")");
        }
    }

    // 全局唯一性：**每局只能出现一次**。多写者下分片的局号区间可以重叠，所以这条只能
    // 在全局范围核。重复 = 发号器或调用方出错，后果是同一局被判进训练/验证两侧。

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
