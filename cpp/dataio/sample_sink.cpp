#include "dataio/sample_sink.h"

#include <algorithm>
#include <stdexcept>

#include "core/fileio.h"

namespace r2048 {

SampleSink::SampleSink(std::string data_dir, BatchMeta meta, uint64_t samples_target,
                       uint64_t shard_samples, bool resume)
    : data_dir_(std::move(data_dir)), meta_(std::move(meta)), samples_target_(samples_target) {
    if (resume) {
        // 续跑：编号接着清单里的局数，所以写入器必须先建好，lane 才能拿到起始编号。
        writer_ = std::make_unique<ShardWriter>(ShardWriter::resume(data_dir_, shard_samples, meta_));
    } else {
        if (!dir_is_usable_as_output(data_dir_)) {
            throw std::runtime_error("输出目录已存在且非空，拒绝覆盖: " + data_dir_ +
                                     "（续跑请用 resume=true，新起一批请换一个目录）");
        }
        ensure_dir(data_dir_);
        writer_ = std::make_unique<ShardWriter>(data_dir_, shard_samples, meta_);
    }
    next_episode_.store(writer_->next_episode());
    next_episode_to_write_ = writer_->next_episode();
    manifest_path_ = data_dir_ + "/manifest.json";
}

uint32_t SampleSink::next_episode() { return next_episode_.fetch_add(1); }

uint32_t SampleSink::peek_next_episode() const { return next_episode_.load(); }

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
    while (true) {
        const auto found = completed_.find(next_episode_to_write_);
        if (found == completed_.end()) break;
        std::vector<Sample> samples = std::move(found->second);
        completed_.erase(found);

        if (samples_target_ > 0 && writer_->samples_written() >= samples_target_) {
            stop_.store(true);
            break;
        }
        if (samples_target_ > 0 && writer_->samples_written() + samples.size() > samples_target_) {
            // 最后一个不完整的局被截断：样本数恰好等于目标条数，且截断只依赖目标，
            // 与并发度、调度顺序无关。不限量（目标为 0）时没有截断可言：一局要么整局落盘，要么不落。
            samples.resize(static_cast<size_t>(samples_target_ - writer_->samples_written()));
            stop_.store(true);
        }
        writer_->add_episode(next_episode_to_write_, samples);
        next_episode_to_write_++;
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
