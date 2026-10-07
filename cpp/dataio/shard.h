#pragma once

#include <cstdint>
#include <unordered_set>
#include <string>
#include <utility>
#include <vector>

#include "core/json.h"
#include "dataio/sample.h"

namespace r2048 {

struct ShardInfo {
    std::string path;  // 相对于数据目录
    uint64_t samples = 0;
    // 本片**写入了几个不同的局**。整批局数 = 各片之和（局号可以有洞，也可能乱序到达，
    // 所以"局数"不能用区间长度、也不能用最高局号推）。
    uint32_t episodes = 0;
    // 本片内容的局号**跨度**（min/max）。它只是给人和校验用的参考：内容可能不连续
    // （同一写者的多局完赛顺序不定），所以区间**重叠是合法的**，别拿它当集合。
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
    uint32_t episodes = 0;
    // 下一次会被发出去的局号（发号器高水位）。局号可以有空洞——发号器按块预留，
    // 写者被杀会留下未用的号——所以"局数"与"最大局号+1"不再相等，两者都要记。
    uint32_t next_episode = 0;
    std::vector<ShardInfo> shards;
};

Json manifest_to_json(const BatchManifest& manifest);
BatchManifest manifest_from_json(const Json& json);

void write_manifest(const std::string& data_dir, const BatchManifest& manifest);
BatchManifest read_manifest(const std::string& data_dir);

// 有序分片写入器：按局号递增接收样本，局满一片就落盘并原子重写一次清单。
//
// 它保证“数据布局只由 (种子, 参数) 决定，与线程数无关”：分片边界只按**局边界**与
// **多个写者可以同时写同一批数据**（多个跑谱进程指向同一个 --out）：
//   * 局号由一个跨进程的发号器（`EpisodeCounter`）统一发放——一局的内容由 (种子, 局号)
//     决定，各写者自己编号会让同一 seed 下不同写者造出同一局（ADR 0007/0020）；
//   * 每个写者写自己的分片文件（名字带写者标识，不可能撞名）；
//   * 清单在**文件锁**下读-改-写地合并：不加锁就会互相覆盖，覆盖等于丢分片（丢数据）。
// 于是"一次运行 = 一个数据批"仍然成立，不需要给每个进程各起一批。
//
// 样本计数切分，因此 1 个 worker 与 16 个 worker 产出逐字节相同的文件。
//
// 清单从此是**增量提交日志**而不是结尾标记：每落一个分片就原子重写一次，它列出的
// 分片才是有效的。于是被 SIGKILL 只会留下“已落盘但未被列出”的孤儿分片——读侧本来
// 就只遍历清单里的分片，所以孤儿不会污染任何东西；续跑会从清单记录的局号接着编号，
// 并把那些孤儿分片按同样的内容重写一遍（第 N 局的内容只由 (种子, 局号) 决定）。
// 跨进程发号器：`<批次目录>/episodes.counter` 里存"下一个可发的号"。
// 每次领一块（`kEpisodeBlock` 个），进程内顺序发完再抢锁领下一块——把"每局抢一次锁"
// 变成"每几千局抢一次锁"。预留出去但没用完的号会留成空洞（写者被杀时必然如此），
// 空洞对读取、划分、训练都没有影响，只影响"局号稠密"这条隐含假设。
class EpisodeCounter {
public:
    static constexpr uint32_t kBlock = 4096;
    // floor：批次里已有的最大局号 + 1（老批次没有计数器文件时从这里开始）。
    //
    // `reset` 区分两种进入方式，这是**逐字节复现**的关键：
    //   * 顺序续跑（只有一个写者）→ reset=true：把计数器拉回 floor。否则块预留过的号
    //     （4096 个）会让续跑从块尾接着编号，于是"续跑 == 一次跑完"这条性质（ADR 0020）
    //     立刻失效——同一批数据里面的局号都不一样了；
    //   * 加入并行批次（join）→ reset=false：只读不写，别把别的写者预留的号抢回来。
    EpisodeCounter(std::string data_dir, uint32_t floor, bool reset);
    uint32_t take();
    uint32_t high_water() const { return limit_; }  // 本块预留到哪（清单里记它，保守但安全）
    // 下一个会被发出去的号（本块内还有剩就是 next_，否则是块尾）。
    uint32_t next_number() const { return next_ < limit_ ? next_ : limit_; }

private:
    void reserve_block();
    std::string path_;
    uint32_t next_ = 0;
    uint32_t limit_ = 0;
};

class ShardWriter {
public:
    // 新起一批：目录必须不存在或为空。
    ShardWriter(std::string data_dir, uint64_t shard_samples, BatchMeta meta);
    // 续跑：读回清单、逐项比对参数、把末片读回缓冲接着填（顺序写者，与并行写者互斥使用）。
    static ShardWriter resume(const std::string& data_dir, uint64_t shard_samples,
                              const BatchMeta& meta);
    // 加入一个**正在被别的写者写**的批次（多进程跑谱）：比对参数、从发号器领号、
    // **不**读回末片（那会把别人还没填满的分片重复写一遍）。
    static ShardWriter join(const std::string& data_dir, uint64_t shard_samples,
                           const BatchMeta& meta);

    // 追加一局。局号由 consume 从发号器领（`take_episode()`），这里只拒绝同一片内重复；
    // 顺序不要求（分片自带局号区间，读取端从记录里的局号还原）。
    void add_episode(uint32_t episode, const std::vector<Sample>& samples);
    // 从发号器领一个局号（一局开始时调用一次）。
    uint32_t take_episode() { return counter_->take(); }

    // 原子重写清单（状态 + 已提交分片）。每落一片自动调用一次。
    void publish(BatchState state);
    BatchState state() const { return state_; }

    uint64_t samples_written() const { return flushed_samples_ + buffered_samples(); }
    // 整批（含别的写者）已提交的局数：局号有空洞，所以按各分片的区间求和，不能用计数。
    uint32_t episodes_written() const;
    // 下一次会被发出去的局号（发号器的高水位）。
    uint32_t next_episode() const { return counter_->high_water(); }
    // 本批已有的最大局号 + 1（join/resume 的起始号）。
    uint32_t episode_floor() const;
    size_t shard_count() const { return shards_.size(); }
    uint64_t shard_samples() const { return shard_samples_; }

    // 落盘最后一片（若非空）、写入终态清单，返回批元数据快照。
    BatchManifest finish(BatchState state);

private:
    void flush_shard();
    void check_episode_order(uint32_t episode) const;
    // 清单更新：在文件锁下读-改-写，把自己新落的分片并进去（不然会覆盖别的写者）。
    // **返回合并后的那份**：返回值与落盘的内容必须同源，否则会出现"两套账"——
    // 实测 finish 报的条数比清单少一个末片（snapshot 只算本写者的分片）。
    BatchManifest merge_into_manifest(BatchState state);
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
    std::unique_ptr<EpisodeCounter> counter_;
    std::string writer_tag_;  // 分片文件名里的写者标识（进程内唯一即可，避免撞名）
    uint32_t my_shard_count_ = 0;  // 本写者落了多少片（只用于分片命名）
    uint32_t batch_episodes_ = 0;  // 整批已提交的局数（各分片区间之和，来自清单合并）
    uint32_t batch_max_episode_plus_one_ = 0;  // 整批已有的最大局号 + 1
    // 续跑时读回来的那个末片：下一次落盘要**覆盖同一个文件**（换名字会让同一段样本
    // 在盘上出现两份，读取端与校验都会看到重复）。
    std::string reuse_shard_path_;
    std::unordered_set<uint32_t> buffer_episodes_;  // 本片已收的局号（查重，O(1)）
    std::vector<ShardInfo> shards_ = {};  // 本写者已提交的分片（其余写者的在清单里）
    BatchState state_ = BatchState::kRunning;
};

}  // namespace r2048
