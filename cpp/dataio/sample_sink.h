#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "core/control.h"
#include "dataio/shard.h"

namespace r2048 {

// 一批数据的写入侧：分片缓冲、按局号顺序落盘、样本目标与终态清单。
//
// 为什么它从引擎里抽出来：多线程搜索 = K 个搜索引擎**各持一份叶子缓存**（天然分片，
// 避开"一个引擎内并行 lane 抢同一个全局缓存锁"那条实测负收益的路），但它们必须产出
// **同一批**数据——一个批次号、一份清单、一套连续局号。所以写入必须是共享的单点：
// 局号从这里的原子计数取，落盘顺序由局号决定（与哪个引擎先跑完无关）。
//
// 第 N 局的内容只由 (种子, 局号) 决定（ADR 0007/0020），所以 K 个引擎共用同一个种子时，
// 它们合起来产出的就是同一套对局，只是每局落在哪个引擎上不同——批次的身份（种子、
// 协议参数）因此仍然准确，不需要给 K 个引擎编造 K 个种子。
class SampleSink {
public:
    // 三种进入方式（与 ShardWriter 一致）：新起 / 顺序续跑 / 加入并行写者。
    enum class Mode { kNew, kResume, kJoin };
    SampleSink(std::string data_dir, BatchMeta meta, uint64_t samples_target, uint64_t shard_samples,
               Mode mode);
    SampleSink(const SampleSink&) = delete;
    SampleSink& operator=(const SampleSink&) = delete;

    // 给一局分配编号。K 个引擎并发调用，靠原子计数保证不重号。
    uint32_t next_episode();
    uint32_t peek_next_episode() const;

    // 一局结束：入队并按局号顺序落盘。编号有空洞时先缓冲，等前一局到齐（顺序由编号决定，
    // 与调度无关——这正是一批数据能被 `--continue` 接续、能与单引擎产出对上的原因）。
    void submit_episode(uint32_t episode, std::vector<Sample> samples);

    uint64_t written_samples() const;
    uint32_t written_episodes() const;
    uint64_t produced_samples() const;
    uint64_t samples_target() const { return samples_target_; }
    size_t shard_count() const;
    // 诊断用：已入队但还没落盘的局数（按序落盘时前一号还没到齐，就会积压在这里）。
    size_t buffered_episodes() const;

    // 停车：谁先看到控制指令谁就把它广播出去（控制文件是一次性消费的，只有一个引擎能读到）。
    void request_stop(ControlCommand command);
    bool stop_requested() const;
    const std::string& requested_stop() const;
    // 样本目标达成（目标为 0 = 不限量，此时永远不会因为样本数停下）。
    bool target_reached() const;

    const std::string& manifest_path() const { return manifest_path_; }
    // 收尾时定下的终态（complete / paused / stopped）。finish() 之前返回 stopped 的初值。
    BatchState state() const { return state_; }
    // 写完剩余分片、落终态清单。**幂等**：多个引擎依次收尾时只有第一次真的写。
    BatchManifest finish();

private:
    // 两个 *locked 版本的存在理由：`finish()` 已经在锁里了，再去调加锁的公开版本
    // 就是自死锁（std::mutex 不可重入）——实测把 C++ 测试挂死过一次。
    void flush_locked();
    bool target_reached_locked() const {
        return samples_target_ > 0 && writer_->samples_written() >= samples_target_;
    }

    std::string data_dir_;
    BatchMeta meta_;
    uint64_t samples_target_;
    std::unique_ptr<ShardWriter> writer_;

    mutable std::mutex mutex_;
    std::map<uint32_t, std::vector<Sample>> completed_;
    uint64_t samples_produced_ = 0;
    std::atomic<bool> stop_{false};
    ControlCommand requested_stop_ = ControlCommand::kNone;
    bool finished_ = false;
    BatchState state_ = BatchState::kStopped;
    std::string manifest_path_;
    BatchManifest manifest_;
};

}  // namespace r2048
