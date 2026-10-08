#include "program/generate.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <semaphore>
#include <stdexcept>
#include <thread>

#include "core/control.h"
#include "core/fileio.h"
#include "core/logging.h"
#include "program/episode.h"

namespace r2048 {

namespace {

std::string join_path(const std::string& dir, const std::string& name) {
    if (dir.empty() || dir.back() == '/') return dir + name;
    return dir + "/" + name;
}

// 批元数据在这里一次性组装：写入器要用它来校验续跑参数，所以不能在结尾才拼。
BatchMeta build_meta(const GenerateConfig& config, uint32_t workers) {
    BatchMeta meta;
    meta.batch_id = config.batch_id.empty() ? std::string("batch") : config.batch_id;
    meta.policy = config.policy;
    meta.simulations = config.simulations;
    // 每一项都必须显式列出：漏写一项就等于把这一项记成默认值，而重放校验会照着
    // 记录跑，于是用它跑出“另一批数据”并报告不一致——错的是记录，不是数据。
    meta.temperature_moves = config.temperature_moves;
    meta.seed = config.seed;
    meta.workers = workers;
    meta.samples_target = config.samples_target;
    meta.model_id = config.model_id;
    meta.params = config.params;
    return meta;
}

}  // namespace

GenerateResult run_generate(const GenerateConfig& config) {
    if (config.shard_samples == 0) throw std::runtime_error("分片样本数必须大于 0");
    if (config.data_dir.empty()) throw std::runtime_error("必须指定输出目录");
    if (config.samples_target > 0 && config.samples_target < config.shard_samples) {
        // 允许，但会在界面上显得奇怪：提示一下让调用方自己确认。
        log_status("提示：样本目标 %llu 小于分片容量 %llu，只会产出一个不完整分片",
                   static_cast<unsigned long long>(config.samples_target),
                   static_cast<unsigned long long>(config.shard_samples));
    }

    const uint32_t workers = config.workers > 0
                                 ? config.workers
                                 : std::max(1u, std::thread::hardware_concurrency());
    const auto slot_capacity = static_cast<std::ptrdiff_t>(2 * workers);

    // 新批要求目录为空；续跑则相反——目录里本来就有半批数据。
    ShardWriter writer = [&] {
        const BatchMeta meta = build_meta(config, workers);
        if (!config.resume) {
            if (!dir_is_usable_as_output(config.data_dir)) {
                throw std::runtime_error(
                    "输出目录已存在且非空，拒绝覆盖: " + config.data_dir +
                    "（续跑请加 --continue，新起一批请换一个目录）");
            }
            ensure_dir(config.data_dir);
            return ShardWriter(config.data_dir, config.shard_samples, meta);
        }
        return ShardWriter::resume(config.data_dir, config.shard_samples, meta);
    }();
    if (config.resume) {
        log_status("续跑：已提交 %llu 条 / %u 局，从第 %u 局接着生成",
                   static_cast<unsigned long long>(writer.samples_written()),
                   writer.episodes_written(), writer.next_episode());
    }

    const std::string control_path =
        config.control_path.empty() ? join_path(config.data_dir, "control") : config.control_path;
    Control control;
    control.set_path(control_path);

    std::mutex mutex;
    std::condition_variable cv;
    std::map<uint32_t, std::vector<Sample>> pending;
    std::counting_semaphore<> slots(slot_capacity);
    std::atomic<bool> stop{false};
    // 续跑从清单的局数接着编号：局号是数据身份的一部分，不能重新从 0 开始。
    std::atomic<uint32_t> next_episode{writer.next_episode()};
    std::atomic<uint32_t> workers_running{workers};
    std::exception_ptr worker_error;
    ControlCommand stop_command = ControlCommand::kNone;

    const double start = monotonic_seconds();

    auto worker_body = [&]() {
        // 每个 worker 持有独立决策器：搜索器内部有可复用的节点池，不能跨线程共享。
        std::unique_ptr<MoveSelector> selector =
            make_selector(config.policy, config.seed, config.simulations, config.temperature_moves,
                          config.greedy_rollout);
        while (true) {
            slots.acquire();
            if (stop.load()) {
                slots.release();
                break;
            }
            const uint32_t episode = next_episode.fetch_add(1);
            std::vector<Sample> samples;
            try {
                EpisodeRecorder recorder(episode, selector->simulations());
                play_episode(config.seed, episode, *selector, &recorder);
                samples = recorder.finish();
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (worker_error == nullptr) worker_error = std::current_exception();
                }
                stop.store(true);
                slots.release();
                cv.notify_all();
                break;
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                pending.emplace(episode, std::move(samples));
            }
            cv.notify_all();
        }
        workers_running.fetch_sub(1);
        cv.notify_all();
    };

    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (uint32_t i = 0; i < workers; i++) threads.emplace_back(worker_body);

    Progress progress("生成样本");
    // 目标为 0 时不设总量：进度只报累计条数，不去编一个假的分母。
    if (config.samples_target > 0) progress.set_total(static_cast<int64_t>(config.samples_target));
    if (config.progress_json) progress.set_json("generate", "samples");

    uint64_t written = writer.samples_written();  // 续跑时从已提交的条数接着算
    uint32_t expected = writer.next_episode();
    bool target_reached = false;
    for (;;) {
        // 叫停只在主循环里看：写入侧是单线程的，在这里决定停车最省事也最可控。
        if (stop_command == ControlCommand::kNone) {
            const ControlCommand command = control.poll();
            if (command != ControlCommand::kNone) {
                stop_command = command;
                log_status("收到 %s：正在停车（进行中的局会走完，已产出的数据会被提交）",
                           control_command_name(command).c_str());
                stop.store(true);
                cv.notify_all();
            }
        }

        std::vector<Sample> samples;
        {
            std::unique_lock<std::mutex> lock(mutex);
            // 停车后仍然要等所有 worker 收工：每个已开局的局都必须写进来，
            // 否则它所属的局号永远不出现，按局号顺序落盘的写入侧就会一直等它。
            cv.wait(lock, [&] {
                return pending.find(expected) != pending.end() ||
                       (stop.load() && workers_running.load() == 0);
            });
            const auto found = pending.find(expected);
            if (found != pending.end()) {
                samples = std::move(found->second);
                pending.erase(found);
            } else {
                break;  // 所有 worker 都已收工，且队列里再没有下一个局号
            }
        }
        slots.release();

        // 目标已达成：这一局（以及后面所有局）一条样本都不需要，直接丢弃。
        // 必须在这里丢弃而不是靠“截断成 0 条”，否则局计数会把这些没有贡献的局算进去，
        // 清单里的局数就会比最后一个分片实际覆盖的局号更大。
        if (config.samples_target > 0 && written >= config.samples_target) {
            stop.store(true);
            cv.notify_all();
            continue;
        }

        // 截断只按目标条数决定，因此写入内容与线程数、线程调度顺序都无关。
        // 不限量时（目标为 0）不截断：一局要么完整落盘，要么下一轮重跑。
        if (config.samples_target > 0 && written + samples.size() > config.samples_target) {
            samples.resize(static_cast<size_t>(config.samples_target - written));
        }
        writer.add_episode(expected, samples);
        written += samples.size();
        expected++;
        progress.update(static_cast<int64_t>(written));
        if (config.samples_target > 0 && written >= config.samples_target) {
            target_reached = true;
            stop.store(true);
            cv.notify_all();
        }
    }
    progress.finish();

    for (std::thread& thread : threads) thread.join();
    if (worker_error != nullptr) std::rethrow_exception(worker_error);

    // 终态：跑满目标就是 complete；否则是被叫停（pause/stop），如实记下是哪一个。
    BatchState state = BatchState::kStopped;
    if (target_reached) {
        state = BatchState::kComplete;
    } else if (stop_command == ControlCommand::kPause) {
        state = BatchState::kPaused;
    }
    const BatchManifest manifest = writer.finish(state);

    GenerateResult result;
    result.samples = manifest.samples;
    result.episodes = manifest.episodes;
    result.seconds = monotonic_seconds() - start;
    result.manifest_path = join_path(config.data_dir, "manifest.json");
    result.state = state;
    return result;
}

}  // namespace r2048
