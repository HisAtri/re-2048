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

}  // namespace

Json manifest_to_json(const BatchManifest& manifest) {
    Json root = Json::object();
    root.set("schema_version", Json::integer(kSampleSchemaVersion));
    root.set("rules_version", Json::integer(kRulesVersion));
    root.set("protocol_version", Json::integer(kProtocolVersion));
    root.set("build_id", Json::str(R2048_BUILD_ID));
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
    // manifest.json 同时是“这一批已经写完”的提交标记：它最后写入，
    // 因此在它出现之前，任何读者都不应当把这批数据当作可用。
    write_text_atomic(data_dir + "/manifest.json", manifest_to_json(manifest).dump(2));
}

BatchManifest read_manifest(const std::string& data_dir) {
    return manifest_from_json(parse_json(read_text_file(data_dir + "/manifest.json")));
}

ShardWriter::ShardWriter(std::string data_dir, uint64_t shard_samples)
    : data_dir_(std::move(data_dir)), shard_samples_(shard_samples) {
    if (shard_samples_ == 0) throw std::runtime_error("分片样本数必须大于 0");
    shards_dir_ = data_dir_ + "/shards";
    ensure_dir(shards_dir_);
    buffer_.reserve(static_cast<size_t>(shard_samples_) * kSampleBytes);
}

void ShardWriter::add_episode(uint32_t episode, const std::vector<Sample>& samples) {
    // 顺序不变式：数据布局必须与线程数无关，所以写入侧强制按局号递增接收。
    if (saw_any_episode_) {
        if (episode <= last_episode_seen_) {
            throw std::runtime_error("局号必须递增：收到第 " + std::to_string(episode) +
                                     " 局，上一局是 " + std::to_string(last_episode_seen_));
        }
    } else if (episode != 0) {
        throw std::runtime_error("第一局的局号必须是 0，实际收到 " + std::to_string(episode));
    }
    saw_any_episode_ = true;
    last_episode_seen_ = episode;

    uint8_t encoded[kSampleBytes];
    for (const Sample& sample : samples) {
        if (sample.episode != episode) throw std::runtime_error("样本的局号与所属局不一致");
        if (buffer_.empty()) buffered_first_episode_ = episode;
        buffered_last_episode_ = episode;
        encode_sample(sample, encoded);
        buffer_.insert(buffer_.end(), encoded, encoded + kSampleBytes);
        samples_written_++;
        if (buffer_.size() == static_cast<size_t>(shard_samples_) * kSampleBytes) flush_shard();
    }
    episodes_written_++;
}

void ShardWriter::flush_shard() {
    if (buffer_.empty()) return;
    const std::string relative = shard_path(shards_.size());
    const std::string absolute = data_dir_ + "/" + relative;
    write_file_atomic(absolute, buffer_.data(), buffer_.size());

    ShardInfo info;
    info.path = relative;
    info.samples = buffer_.size() / kSampleBytes;
    info.first_episode = buffered_first_episode_;
    info.last_episode = buffered_last_episode_;
    info.sha256 = sha256_hex(buffer_.data(), buffer_.size());
    shards_.push_back(std::move(info));
    buffer_.clear();
}

BatchManifest ShardWriter::finish(const BatchMeta& meta) {
    flush_shard();
    BatchManifest manifest;
    manifest.meta = meta;
    manifest.meta.shard_samples = shard_samples_;
    manifest.samples = samples_written_;
    manifest.episodes = episodes_written_;
    manifest.shards = shards_;
    return manifest;
}

}  // namespace r2048
