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
    // 0 表示不限量：跑到被外部叫停为止（见 core/control.h）。
    uint64_t samples_target = 0;
    std::string model_id;  // 非 NN 策略为空
    // 产出这批数据的构建身份（编译期常量）。换二进制续跑会让前后两半出自不同的搜索实现，
    // 因此它属于续跑要比对的字段。
    std::string build_id;
    // 命令行与调用方的原样记录（含非协议项，如 --workers、--out）。用于人看，
    // 也用于续跑时的协议项比对（只比对 kProtocolParams 里声明过的名字）。
    std::vector<std::pair<std::string, std::string>> params;
};

// 批次状态。它不是装饰：续跑只允许发生在非 complete 的批次上，
// 而“这批到底跑完了没有”此前无处可查（旧实现只靠 manifest 是否存在来表示）。
enum class BatchState { kRunning, kPaused, kStopped, kComplete };
std::string batch_state_name(BatchState state);
BatchState batch_state_from_name(const std::string& name);

// 分片切分规则。它决定“分片边界在哪里”，因此换规则之后旧批次无法被重放比对——
// 记录在清单里，让 verify 能区分“数据错了”与“切分规则变了”。
//   1：攒满容量即切（旧实现；一局可能横跨两个分片）
//   2：只在局边界上切（当前实现；末片永远结束在一局的末尾）
constexpr int kShardLayout = 2;

struct BatchManifest {
    BatchMeta meta;
    BatchState state = BatchState::kComplete;
    int shard_layout = kShardLayout;
    uint64_t samples = 0;   // 已提交（已落盘且列在 shards 里）的条数
    uint32_t episodes = 0;  // 已提交的局数
    std::vector<ShardInfo> shards;
};

Json manifest_to_json(const BatchManifest& manifest);
BatchManifest manifest_from_json(const Json& json);

void write_manifest(const std::string& data_dir, const BatchManifest& manifest);
BatchManifest read_manifest(const std::string& data_dir);

// 有序分片写入器：按局号递增接收样本，局满一片就落盘并原子重写一次清单。
//
// 它保证“数据布局只由 (种子, 参数) 决定，与线程数无关”：分片边界只按**局边界**与
// 样本计数切分，因此 1 个 worker 与 16 个 worker 产出逐字节相同的文件。
//
// 清单从此是**增量提交日志**而不是结尾标记：每落一个分片就原子重写一次，它列出的
// 分片才是有效的。于是被 SIGKILL 只会留下“已落盘但未被列出”的孤儿分片——读侧本来
// 就只遍历清单里的分片，所以孤儿不会污染任何东西；续跑会从清单记录的局号接着编号，
// 并把那些孤儿分片按同样的内容重写一遍（第 N 局的内容只由 (种子, 局号) 决定）。
class ShardWriter {
public:
    // 新起一批：目录必须不存在或为空。
    ShardWriter(std::string data_dir, uint64_t shard_samples, BatchMeta meta);
    // 续跑：读回清单、逐项比对参数、必要时把末片读回缓冲接着填。
    // 参数不一致（种子/策略/模拟次数/模型/构建/分片容量/协议参数）直接拒绝并指名道姓，
    // 因为“参数变了”意味着这是另一个来源，应该新起一批，而不是缝在一起。
    static ShardWriter resume(const std::string& data_dir, uint64_t shard_samples,
                              const BatchMeta& meta);

    // 必须按局号连续递增调用；跳号或回退属于调用方 bug，直接抛错。
    void add_episode(uint32_t episode, const std::vector<Sample>& samples);

    // 原子重写清单（状态 + 已提交分片）。每落一片自动调用一次。
    void publish(BatchState state);
    BatchState state() const { return state_; }

    uint64_t samples_written() const { return flushed_samples_ + buffered_samples(); }
    uint32_t episodes_written() const { return episodes_written_; }
    uint32_t next_episode() const { return episodes_written_; }
    size_t shard_count() const { return shards_.size(); }
    uint64_t shard_samples() const { return shard_samples_; }

    // 落盘最后一片（若非空）、写入终态清单，返回批元数据快照。
    BatchManifest finish(BatchState state);

private:
    void flush_shard();
    void check_episode_order(uint32_t episode) const;
    BatchManifest snapshot(BatchState state) const;
    uint64_t buffered_samples() const { return buffer_.size() / kSampleBytes; }
    uint64_t capacity_bytes() const { return shard_samples_ * kSampleBytes; }

    std::string data_dir_;
    std::string shards_dir_;
    uint64_t shard_samples_;
    BatchMeta meta_;
    std::vector<uint8_t> buffer_;
    uint32_t buffered_first_episode_ = 0;
    uint32_t buffered_last_episode_ = 0;
    uint64_t flushed_samples_ = 0;   // 已列在 shards_ 里的条数
    uint32_t episodes_written_ = 0;  // 已完整交给写入器的局数
    std::vector<ShardInfo> shards_;
    BatchState state_ = BatchState::kRunning;
};

}  // namespace r2048
