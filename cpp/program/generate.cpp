#include "program/generate.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <semaphore>
#include <stdexcept>
#include <thread>

#include "core/fileio.h"
#include "core/logging.h"
#include "program/episode.h"

namespace r2048 {

namespace {

std::string join_path(const std::string& dir, const std::string& name) {
    if (dir.empty() || dir.back() == '/') return dir + name;
    return dir + "/" + name;
}

}  // namespace

GenerateResult run_generate(const GenerateConfig& config) {
    if (config.samples_target == 0) throw std::runtime_error("样本目标条数必须大于 0");
    if (config.shard_samples == 0) throw std::runtime_error("分片样本数必须大于 0");
    if (config.data_dir.empty()) throw std::runtime_error("必须指定输出目录");
    if (config.samples_target < config.shard_samples) {
        // 允许，但会在界面上显得奇怪：提示一下让调用方自己确认。
        log_status("提示：样本目标 %llu 小于分片容量 %llu，只会产出一个不完整分片",
                   static_cast<unsigned long long>(config.samples_target),
                   static_cast<unsigned long long>(config.shard_samples));
    }
    if (!dir_is_usable_as_output(config.data_dir)) {
        throw std::runtime_error("输出目录已存在且非空，拒绝覆盖: " + config.data_dir +
                                 "（跑谱批按“整批要么完整、要么重来”管理，请换一个目录或先清理）");
    }
    ensure_dir(config.data_dir);

    const uint32_t workers = config.workers > 0
                                 ? config.workers
                                 : std::max(1u, std::thread::hardware_concurrency());
    const auto slot_capacity = static_cast<std::ptrdiff_t>(2 * workers);

    ShardWriter writer(config.data_dir, config.shard_samples);

    std::mutex mutex;
    std::condition_variable cv;
    std::map<uint32_t, std::vector<Sample>> pending;
    std::counting_semaphore<> slots(slot_capacity);
    std::atomic<bool> stop{false};
    std::atomic<uint32_t> next_episode{0};
    std::atomic<uint32_t> workers_running{workers};
    std::exception_ptr worker_error;

    const double start = monotonic_seconds();

    auto worker_body = [&]() {
        // 每个 worker 持有独立决策器：搜索器内部有可复用的节点池，不能跨线程共享。
        std::unique_ptr<MoveSelector> selector =
            make_selector(config.policy, config.seed, config.simulations, config.temperature_moves);
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
    progress.set_total(static_cast<int64_t>(config.samples_target));

    uint64_t written = 0;
    uint32_t expected = 0;
    for (;;) {
        std::vector<Sample> samples;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] {
                return pending.find(expected) != pending.end() || stop.load() ||
                       workers_running.load() == 0;
            });
            const auto found = pending.find(expected);
            if (found != pending.end()) {
                samples = std::move(found->second);
                pending.erase(found);
            } else {
                break;  // 目标已达成或生产者出错
            }
        }
        slots.release();

        // 目标已达成：这一局（以及后面所有局）一条样本都不需要，直接丢弃。
        // 必须在这里丢弃而不是靠“截断成 0 条”，否则局计数会把这些没有贡献的局算进去，
        // 清单里的局数就会比最后一个分片实际覆盖的局号更大。
        if (written >= config.samples_target) {
            stop.store(true);
            cv.notify_all();
            continue;
        }

        // 截断只按目标条数决定，因此写入内容与线程数、线程调度顺序都无关。
        if (written + samples.size() > config.samples_target) {
            samples.resize(static_cast<size_t>(config.samples_target - written));
        }
        writer.add_episode(expected, samples);
        written += samples.size();
        expected++;
        progress.update(static_cast<int64_t>(written));
        if (written >= config.samples_target) {
            stop.store(true);
            cv.notify_all();
        }
    }
    progress.finish();

    for (std::thread& thread : threads) thread.join();
    if (worker_error != nullptr) std::rethrow_exception(worker_error);

    BatchMeta meta;
    meta.batch_id = config.batch_id.empty() ? std::string("batch") : config.batch_id;
    meta.policy = config.policy;
    meta.simulations = config.simulations;
    meta.seed = config.seed;
    meta.workers = workers;
    meta.samples_target = config.samples_target;
    meta.model_id = config.model_id;
    meta.params = config.params;

    BatchManifest manifest = writer.finish(meta);
    const std::string manifest_path = join_path(config.data_dir, "manifest.json");
    write_manifest(config.data_dir, manifest);

    GenerateResult result;
    result.samples = manifest.samples;
    result.episodes = manifest.episodes;
    result.seconds = monotonic_seconds() - start;
    result.manifest_path = manifest_path;
    return result;
}

}  // namespace r2048
