#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/json.h"
#include "dataio/sample.h"

namespace r2048 {

struct ShardInfo {
    std::string path;  // 相对于数据目录
    uint64_t samples = 0;
    uint32_t first_episode = 0;
    uint32_t last_episode = 0;
    std::string sha256;
};

// 一个数据批的“来源记录”：谁产出的、用什么参数产出的。
// 这批数据能不能和别的批次混着训练，全靠这些字段判断，所以它们必须落在文件里，
// 而不是留在某次命令行历史中。
struct BatchMeta {
    std::string batch_id;
    std::string policy;  // random / mcts / nn
    uint32_t simulations = 0;
    // 开局若干步按根访问分布采样：它直接影响产出数据的行为策略，因此属于协议的一部分，
    // 必须记录，否则重放校验会因为默认值不同而“验证失败”。
    uint32_t temperature_moves = 0;
    uint64_t seed = 0;
    uint64_t shard_samples = 0;
    uint32_t workers = 0;
    uint64_t samples_target = 0;
    std::string model_id;  // 非 NN 策略为空
    std::vector<std::pair<std::string, std::string>> params;
};

struct BatchManifest {
    BatchMeta meta;
    uint64_t samples = 0;
    uint32_t episodes = 0;
    std::vector<ShardInfo> shards;
};

Json manifest_to_json(const BatchManifest& manifest);
BatchManifest manifest_from_json(const Json& json);

void write_manifest(const std::string& data_dir, const BatchManifest& manifest);
BatchManifest read_manifest(const std::string& data_dir);

// 有序分片写入器：按局号递增接收样本，攒满一片就落盘并登记哈希。
//
// 它保证“数据布局只由 (种子, 参数) 决定，与线程数无关”：分片边界只按样本计数
// 切分，因此 1 个 worker 与 16 个 worker 产出逐字节相同的文件。
class ShardWriter {
public:
    ShardWriter(std::string data_dir, uint64_t shard_samples);

    // 必须按局号递增调用；局号回退属于调用方 bug，直接抛错。
    void add_episode(uint32_t episode, const std::vector<Sample>& samples);

    uint64_t samples_written() const { return samples_written_; }
    uint32_t episodes_written() const { return episodes_written_; }
    size_t shard_count() const { return shards_.size(); }

    // 落盘最后一片（若非空），返回尚未写入文件的批元数据。
    BatchManifest finish(const BatchMeta& meta);

private:
    void flush_shard();

    std::string data_dir_;
    std::string shards_dir_;
    uint64_t shard_samples_;
    std::vector<uint8_t> buffer_;
    uint32_t buffered_first_episode_ = 0;  // 当前缓冲第一条记录的局号
    uint32_t buffered_last_episode_ = 0;   // 当前缓冲最后一条记录的局号
    uint32_t last_episode_seen_ = 0;
    bool saw_any_episode_ = false;
    uint64_t samples_written_ = 0;
    uint32_t episodes_written_ = 0;
    std::vector<ShardInfo> shards_;
};

}  // namespace r2048
