#include "dataio/sample_sink.h"

#include <algorithm>
#include <stdexcept>

#include "core/fileio.h"

namespace r2048 {

SampleSink::SampleSink(std::string data_dir, BatchMeta meta, uint64_t samples_target,
                       uint64_t shard_samples, Mode mode)
    : data_dir_(std::move(data_dir)), meta_(std::move(meta)), samples_target_(samples_target) {
    if (mode == Mode::kResume) {
        // 顺序续跑：编号接着清单里的最大局号，所以写入器必须先建好，lane 才能拿到起始编号。
        writer_ = std::make_unique<ShardWriter>(ShardWriter::resume(data_dir_, shard_samples, meta_));
    } else if (mode == Mode::kJoin) {
        // 加入并行写者：与 resume 的区别是**不读回末片**（它可能正被别的写者填着）。
        writer_ = std::make_unique<ShardWriter>(ShardWriter::join(data_dir_, shard_samples, meta_));
    } else {
        if (!dir_is_usable_as_output(data_dir_)) {
            throw std::runtime_error("输出目录已存在且非空，拒绝覆盖: " + data_dir_ +
                                     "（续跑请用 resume=true，新起一批请换一个目录）");
        }
        ensure_dir(data_dir_);
        writer_ = std::make_unique<ShardWriter>(data_dir_, shard_samples, meta_);
    }
    manifest_path_ = data_dir_ + "/manifest.json";
}

uint32_t SampleSink::next_episode() { return writer_->take_episode(); }

uint32_t SampleSink::peek_next_episode() const { return writer_->next_episode(); }

void SampleSink::submit_episode(uint32_t episode, std::vector<Sample> samples) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        samples_produced_ += samples.size();
        completed_[episode] = std::move(samples);
        if (samples_target_ > 0 && samples_produced_ >= samples_target_) stop_.store(true);
        flush_locked();
    }
}

void SampleSink::flush_locked() {
    // 把**已完成的局**按局号排序后落盘，不等"下一个号"。
    //
    // 旧实现（单写者时代）会等 `next_episode_to_write_` 到齐才落盘，好处是每个分片正好是
    // 一段连续局号。但局号现在由**跨进程**发号器发放，某个号可能属于另一个写者、永远不会
    // 送到这里——等它就等于永久卡住。所以：分片按"自己收到的局号区间"落盘，区间可以有洞，
    // 读取端从每条记录的局号还原（分片自带 first/last_episode），顺序不再参与语义。
    while (!completed_.empty()) {
        auto it = completed_.begin();  // std::map 有序：每次都取最小的那一局
        const uint32_t episode = it->first;
        std::vector<Sample> samples = std::move(it->second);
        completed_.erase(it);

        if (samples_target_ > 0 && writer_->samples_written() >= samples_target_) {
            stop_.store(true);
            break;
        }
        if (samples_target_ > 0 && writer_->samples_written() + samples.size() > samples_target_) {
            // 目标边界上那一局被截断：总数恰好等于目标。多写者下"哪一局被截断"取决于落盘
            // 顺序（不再唯一），但**条数只由目标决定**——训练看到的总量是确定的。
            samples.resize(static_cast<size_t>(samples_target_ - writer_->samples_written()));
            stop_.store(true);
        }
        writer_->add_episode(episode, samples);
        if (samples_target_ > 0 && writer_->samples_written() >= samples_target_) stop_.store(true);
    }
}

uint64_t SampleSink::written_samples() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return writer_->samples_written();
}

uint32_t SampleSink::written_episodes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return writer_->episodes_written();
}

uint64_t SampleSink::produced_samples() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return samples_produced_;
}

size_t SampleSink::shard_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return writer_->shard_count();
}

size_t SampleSink::buffered_episodes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return completed_.size();
}

void SampleSink::request_stop(ControlCommand command) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // 只在第一次记录原因：pause 与 stop 都可能被不同引擎先后看到，
        // 而清单状态只有一个，先到的那个才是这次停车的原因。
        if (requested_stop_ == ControlCommand::kNone) requested_stop_ = command;
    }
    stop_.store(true);
}

bool SampleSink::stop_requested() const { return stop_.load(); }

const std::string& SampleSink::requested_stop() const {
    static const std::string kNone = "none";
    static const std::string kPause = "pause";
    static const std::string kStop = "stop";
    switch (requested_stop_) {
        case ControlCommand::kPause: return kPause;
        case ControlCommand::kStop: return kStop;
        default: return kNone;
    }
}

bool SampleSink::target_reached() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return target_reached_locked();
}

BatchManifest SampleSink::finish() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (finished_) return manifest_;
    stop_.store(true);
    flush_locked();

    // 终态：跑满目标是 complete，否则如实记下这次是怎么停的（pause 还是 stop）。
    // “停在哪”直接决定这批数据能不能被接着写。
    BatchState state = BatchState::kStopped;
    if (target_reached_locked()) {
        state = BatchState::kComplete;
    } else if (requested_stop_ == ControlCommand::kPause) {
        state = BatchState::kPaused;
    }
    manifest_ = writer_->finish(state);
    state_ = state;
    finished_ = true;
    return manifest_;
}

}  // namespace r2048
