#include "dataio/shard.h"

#include <cstdio>
#include <stdexcept>

#include "core/fileio.h"
#include "core/hash.h"
#include "r2048_build_id.h"  // 生成文件：batches 必须记录产出它们的构建身份

namespace r2048 {

namespace {

std::string shard_path(size_t index) {
    char name[64];
    std::snprintf(name, sizeof(name), "shards/shard-%06zu.bin", index);
    return name;
}

std::string join_path(const std::string& dir, const std::string& name) {
    if (dir.empty() || dir.back() == '/') return dir + name;
    return dir + "/" + name;
}

void require_equal(const std::string& name, const std::string& existing, const std::string& requested) {
    if (existing == requested) return;
    throw std::runtime_error("续跑参数与既有数据不一致：" + name + " 原为 " +
                            (existing.empty() ? "(空)" : existing) + "，本次为 " +
                            (requested.empty() ? "(空)" : requested) +
                            "。参数变了就是另一个来源，请新起一批而不是缝在一起");
}

void require_equal_u64(const std::string& name, uint64_t existing, uint64_t requested) {
    if (existing == requested) return;
    require_equal(name, std::to_string(existing), std::to_string(requested));
}

const std::string* find_param(const std::vector<std::pair<std::string, std::string>>& params,
                              const std::string& key) {
    for (const auto& entry : params) {
        if (entry.first == key) return &entry.second;
    }
    return nullptr;
}

// 清单 params 里**属于协议**的键：它们会改变搜索或前向的行为，因此续跑必须一致。
// 不在这个列表里的键（--samples、--workers、--out、--control 等）可以合法变化，
// 它们只是“当时是怎么敲的命令行”的记录。
const char* const kProtocolParams[] = {
    "c_puct", "lanes", "in_flight_per_lane", "temperature", "cache_entries", "max_batch", "device",
    // 搜索线程数会改变 lane 的调度顺序 → 改变每批请求的组成 → 改变逐局结果（ADR 0017），
    // 所以和 lanes / in_flight 一样属于协议参数，进清单、也进数据来源身份（ADR 0021）。
    "search_threads",
};

void compare_protocol_params(const BatchMeta& existing, const BatchMeta& requested) {
    for (const char* name : kProtocolParams) {
        const std::string* before = find_param(existing.params, name);
        const std::string* after = find_param(requested.params, name);
        if (before == nullptr && after == nullptr) continue;
        // 一侧记了、另一侧没记，同样是“不是同一套参数”：这正是缺字段必须报错的场合。
        require_equal(name, before == nullptr ? "" : *before, after == nullptr ? "" : *after);
    }
}

// 续跑的参数闸门：逐项比对，直接把不同的那一项名字说出来。
// 刻意不做“摘要比对”——摘要只能告诉你“不一致”，而这里要的是“哪一项变了”。
void compare_resume_meta(const BatchMeta& existing, const BatchMeta& requested, uint64_t shard_samples) {
    require_equal("policy", existing.policy, requested.policy);
    require_equal("model_id", existing.model_id, requested.model_id);
    // 产出者身份是写入器自己填的编译期常量：换二进制意味着前后两半出自不同的搜索实现。
    require_equal("build_id", existing.build_id, std::string(R2048_BUILD_ID));
    require_equal_u64("simulations", existing.simulations, requested.simulations);
    require_equal_u64("temperature_moves", existing.temperature_moves, requested.temperature_moves);
    require_equal_u64("seed", existing.seed, requested.seed);
    require_equal_u64("shard_samples", existing.shard_samples, shard_samples);
    compare_protocol_params(existing, requested);
}

std::vector<uint8_t> read_binary(const std::string& path) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) throw std::runtime_error("无法读取文件: " + path);
    std::vector<uint8_t> data;
    uint8_t chunk[1 << 16];
    size_t read = 0;
    while ((read = std::fread(chunk, 1, sizeof(chunk), file)) > 0) {
        data.insert(data.end(), chunk, chunk + read);
    }
    const bool failed = std::ferror(file) != 0;
    std::fclose(file);
    if (failed) throw std::runtime_error("读取文件失败: " + path);
    return data;
}

}  // namespace

std::string batch_state_name(BatchState state) {
    switch (state) {
        case BatchState::kRunning:
            return "running";
        case BatchState::kPaused:
            return "paused";
        case BatchState::kStopped:
            return "stopped";
        case BatchState::kComplete:
            return "complete";
    }
    return "complete";
}

BatchState batch_state_from_name(const std::string& name) {
    if (name == "running") return BatchState::kRunning;
    if (name == "paused") return BatchState::kPaused;
    if (name == "stopped") return BatchState::kStopped;
    if (name == "complete") return BatchState::kComplete;
    throw std::runtime_error("无法识别的批次状态: " + name);
}

Json manifest_to_json(const BatchManifest& manifest) {
    Json root = Json::object();
    root.set("schema_version", Json::integer(kSampleSchemaVersion));
    root.set("rules_version", Json::integer(kRulesVersion));
    root.set("protocol_version", Json::integer(kProtocolVersion));
    root.set("build_id", Json::str(manifest.meta.build_id));
    root.set("shard_layout", Json::integer(manifest.shard_layout));
    root.set("batch_id", Json::str(manifest.meta.batch_id));
    root.set("policy", Json::str(manifest.meta.policy));
    root.set("simulations", Json::integer(manifest.meta.simulations));
    root.set("temperature_moves", Json::integer(manifest.meta.temperature_moves));
    root.set("seed", Json::integer(static_cast<int64_t>(manifest.meta.seed)));
    root.set("model_id", Json::str(manifest.meta.model_id));
    root.set("record_bytes", Json::integer(static_cast<int64_t>(kSampleBytes)));
    root.set("shard_samples", Json::integer(manifest.meta.shard_samples));
    root.set("samples_target", Json::integer(static_cast<int64_t>(manifest.meta.samples_target)));
    root.set("samples", Json::integer(static_cast<int64_t>(manifest.samples)));
    root.set("episodes", Json::integer(manifest.episodes));
    root.set("workers", Json::integer(manifest.meta.workers));
    root.set("state", Json::str(batch_state_name(manifest.state)));

    Json params = Json::object();
    for (const auto& entry : manifest.meta.params) params.set(entry.first, Json::str(entry.second));
    root.set("params", std::move(params));

    Json shards = Json::array();
    for (const ShardInfo& shard : manifest.shards) {
        Json entry = Json::object();
        entry.set("path", Json::str(shard.path));
        entry.set("samples", Json::integer(static_cast<int64_t>(shard.samples)));
        entry.set("first_episode", Json::integer(shard.first_episode));
        entry.set("last_episode", Json::integer(shard.last_episode));
        entry.set("sha256", Json::str(shard.sha256));
        shards.push(std::move(entry));
    }
    root.set("shards", std::move(shards));
    return root;
}

BatchManifest manifest_from_json(const Json& json) {
    BatchManifest manifest;
    manifest.meta.batch_id = json.at("batch_id").as_string();
    manifest.meta.policy = json.at("policy").as_string();
    manifest.meta.simulations = static_cast<uint32_t>(json.at("simulations").as_int());
    manifest.meta.temperature_moves = static_cast<uint32_t>(json.at("temperature_moves").as_int());
    manifest.meta.seed = static_cast<uint64_t>(json.at("seed").as_int());
    manifest.meta.model_id = json.at("model_id").as_string();
    manifest.meta.shard_samples = static_cast<uint64_t>(json.at("shard_samples").as_int());
    manifest.meta.samples_target = static_cast<uint64_t>(json.at("samples_target").as_int());
    manifest.meta.workers = static_cast<uint32_t>(json.at("workers").as_int());
    manifest.samples = static_cast<uint64_t>(json.at("samples").as_int());
    manifest.episodes = static_cast<uint32_t>(json.at("episodes").as_int());
    // 这两个字段是“可增量提交”之后才有的。旧清单没有它们，而旧实现只在结尾写一次清单
    // （写到即完整），因此按 complete + 旧切分规则解释是准确的，不是兜底猜测。
    manifest.state = json.has("state") ? batch_state_from_name(json.at("state").as_string())
                                       : BatchState::kComplete;
    manifest.shard_layout = json.has("shard_layout") ? static_cast<int>(json.at("shard_layout").as_int()) : 1;
    // build_id 同样是后加的字段；缺它说明清单来自更早的实现，留空即可（续跑会因此拒绝，
    // 这正是想要的：没有产出者身份就不能续写）。
    manifest.meta.build_id = json.has("build_id") ? json.at("build_id").as_string() : "";

    for (const auto& entry : json.at("params").items()) {
        manifest.meta.params.emplace_back(entry.first, entry.second.as_string());
    }
    for (const Json& entry : json.at("shards").elements()) {
        ShardInfo shard;
        shard.path = entry.at("path").as_string();
        shard.samples = static_cast<uint64_t>(entry.at("samples").as_int());
        shard.first_episode = static_cast<uint32_t>(entry.at("first_episode").as_int());
        shard.last_episode = static_cast<uint32_t>(entry.at("last_episode").as_int());
        shard.sha256 = entry.at("sha256").as_string();
        manifest.shards.push_back(std::move(shard));
    }

    // 版本不匹配意味着数据是另一套语义产出的，不允许继续使用。
    const int64_t schema = json.at("schema_version").as_int();
    const int64_t rules = json.at("rules_version").as_int();
    if (schema != kSampleSchemaVersion) {
        throw std::runtime_error("数据批的记录格式版本不匹配: " + std::to_string(schema) + " != " +
                                 std::to_string(kSampleSchemaVersion));
    }
    if (rules != kRulesVersion) {
        throw std::runtime_error("数据批的规则版本不匹配: " + std::to_string(rules) + " != " +
                                 std::to_string(kRulesVersion));
    }
    return manifest;
}

void write_manifest(const std::string& data_dir, const BatchManifest& manifest) {
    // 原子重写：清单现在会被反复改写（每落一片一次），而读者可能在任意时刻读它。
    // “存在即完整”这条性质全靠 tmp + rename 撑着。
    write_text_atomic(data_dir + "/manifest.json", manifest_to_json(manifest).dump(2));
}

BatchManifest read_manifest(const std::string& data_dir) {
    return manifest_from_json(parse_json(read_text_file(data_dir + "/manifest.json")));
}

ShardWriter::ShardWriter(std::string data_dir, uint64_t shard_samples, BatchMeta meta)
    : data_dir_(std::move(data_dir)), shard_samples_(shard_samples), meta_(std::move(meta)) {
    if (shard_samples_ == 0) throw std::runtime_error("分片样本数必须大于 0");
    // 产出者身份由写入器自己填：它按定义就是产出这批数据的构建，调用方不需要（也不应该）
    // 各自去读一遍编译期常量。
    meta_.build_id = R2048_BUILD_ID;
    shards_dir_ = data_dir_ + "/shards";
    ensure_dir(shards_dir_);
    buffer_.reserve(static_cast<size_t>(shard_samples_) * kSampleBytes);
}

ShardWriter ShardWriter::resume(const std::string& data_dir, uint64_t shard_samples,
                                const BatchMeta& meta) {
    const BatchManifest existing = read_manifest(data_dir);
    if (existing.state == BatchState::kComplete) {
        throw std::runtime_error("这一批已经跑完（state=complete）：追加数据请新起一批"
                                 "（多来源训练见 ADR 0021，不需要把两批缝成一批）");
    }
    if (existing.shard_layout != kShardLayout) {
        throw std::runtime_error("该批使用旧的分片切分规则（shard_layout=" +
                                 std::to_string(existing.shard_layout) + "，当前为 " +
                                 std::to_string(kShardLayout) +
                                 "）：续写会让同一批数据出现两种分片布局，请新起一批");
    }
    compare_resume_meta(existing.meta, meta, shard_samples);

    ShardWriter writer(data_dir, shard_samples, meta);
    writer.flushed_samples_ = existing.samples;
    writer.episodes_written_ = existing.episodes;
    writer.shards_ = existing.shards;

    // 末片不满 = 上一次是在停车点落盘的。把它读回缓冲继续填，而不是新起一片：
    // 这样续跑最终产出的分片与“一次跑完”的分片逐字节相同（强判据，见 ADR 0020）。
    if (!writer.shards_.empty()) {
        const ShardInfo last = writer.shards_.back();
        if (last.samples < shard_samples) {
            const std::vector<uint8_t> bytes = read_binary(join_path(data_dir, last.path));
            if (bytes.size() != last.samples * kSampleBytes) {
                throw std::runtime_error("末分片 " + last.path + " 的字节数与清单不符，拒绝续写");
            }
            const Sample first = decode_sample(bytes.data());
            const Sample final = decode_sample(bytes.data() + bytes.size() - kSampleBytes);
            if (first.episode != last.first_episode || final.episode != last.last_episode) {
                throw std::runtime_error("末分片 " + last.path + " 的局号与清单不符，拒绝续写");
            }
            writer.buffer_.assign(bytes.begin(), bytes.end());
            writer.buffered_first_episode_ = last.first_episode;
            writer.buffered_last_episode_ = last.last_episode;
            writer.shards_.pop_back();  // 它还没填满，等填满后按同样的内容重写
            writer.flushed_samples_ -= last.samples;
        }
    }
    return writer;
}

void ShardWriter::check_episode_order(uint32_t episode) const {
    if (episode != episodes_written_) {
        throw std::runtime_error("局号必须从 " + std::to_string(episodes_written_) +
                                 " 开始连续递增，实际收到第 " + std::to_string(episode) +
                                 " 局（跳号会让清单的局数与分片内容对不上）");
    }
}

void ShardWriter::add_episode(uint32_t episode, const std::vector<Sample>& samples) {
    check_episode_order(episode);

    uint8_t encoded[kSampleBytes];
    for (const Sample& sample : samples) {
        if (sample.episode != episode) throw std::runtime_error("样本的局号与所属局不一致");
        if (buffer_.empty()) buffered_first_episode_ = episode;
        buffered_last_episode_ = episode;
        encode_sample(sample, encoded);
        buffer_.insert(buffer_.end(), encoded, encoded + kSampleBytes);
    }
    episodes_written_++;

    // 分片只在**局边界**上切分（攒满容量之后的第一局末尾）。这条规则换来两件事：
    //   * 末片的 last_episode 永远是完整的一局，清单里的 episodes 恒等于 last_episode+1，
    //     于是续跑从 manifest.episodes 接着编号即可，不存在“半局”这种状态；
    //   * 切分点只由 (种子, 参数, 样本流) 决定，与 worker 数、暂停次数都无关。
    // 代价是分片会略微超过容量（最多多出一局的样本），这对读取侧完全透明。
    if (buffer_.size() >= static_cast<size_t>(capacity_bytes())) flush_shard();
}

void ShardWriter::flush_shard() {
    if (buffer_.empty()) return;
    const std::string relative = shard_path(shards_.size());
    const std::string absolute = join_path(data_dir_, relative);
    write_file_atomic(absolute, buffer_.data(), buffer_.size());

    ShardInfo info;
    info.path = relative;
    info.samples = buffer_.size() / kSampleBytes;
    info.first_episode = buffered_first_episode_;
    info.last_episode = buffered_last_episode_;
    info.sha256 = sha256_hex(buffer_.data(), buffer_.size());
    shards_.push_back(std::move(info));
    flushed_samples_ += buffer_.size() / kSampleBytes;
    buffer_.clear();
    // 每落一片就提交一次清单：清单从此是提交日志，被 SIGKILL 只会留下未被列出的孤儿分片。
    publish(state_);
}

BatchManifest ShardWriter::snapshot(BatchState state) const {
    BatchManifest manifest;
    manifest.meta = meta_;
    manifest.meta.shard_samples = shard_samples_;
    manifest.state = state;
    manifest.shard_layout = kShardLayout;
    // 已提交的内容从定义上只包含分片里的记录：缓冲里的数据还没落盘，不能算数
    // （否则崩溃之后清单会声称存在一批读不出来的数据）。
    manifest.samples = flushed_samples_;
    manifest.episodes = episodes_written_;
    manifest.shards = shards_;
    return manifest;
}

void ShardWriter::publish(BatchState state) {
    state_ = state;
    write_manifest(data_dir_, snapshot(state));
}

BatchManifest ShardWriter::finish(BatchState state) {
    flush_shard();  // 末尾那一段不满的数据在这里落盘（它一定结束在一局的末尾）
    publish(state);
    return snapshot(state);
}

}  // namespace r2048
