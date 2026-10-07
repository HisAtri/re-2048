#include "program/evaluate.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "core/fileio.h"
#include "core/json.h"
#include "core/logging.h"
#include "core/version.h"
#include "program/episode.h"

namespace r2048 {

EvaluateResult run_evaluate(const EvaluateConfig& config) {
    if (config.games == 0) throw std::runtime_error("对局数必须大于 0");
    const uint32_t workers = config.workers > 0
                                 ? config.workers
                                 : std::max(1u, std::thread::hardware_concurrency());
    const double start = monotonic_seconds();

    std::vector<GameOutcome> outcomes(config.games);
    std::atomic<uint64_t> next_game{0};
    std::atomic<uint64_t> finished{0};
    std::exception_ptr worker_error;

    auto worker_body = [&]() {
        std::unique_ptr<MoveSelector> selector =
            make_selector(config.policy, config.seed, config.simulations, config.temperature_moves);
        while (true) {
            const uint64_t index = next_game.fetch_add(1);
            if (index >= config.games) break;
            try {
                const EpisodeOutcome outcome =
                    play_episode(config.seed, static_cast<uint32_t>(index), *selector, nullptr);
                GameOutcome record;
                record.episode = static_cast<uint32_t>(index);
                record.score = outcome.total_score;
                record.max_exponent = outcome.max_exponent;
                record.steps = outcome.steps;
                outcomes[index] = record;
            } catch (...) {
                worker_error = std::current_exception();
                next_game.store(config.games);
                break;
            }
            finished.fetch_add(1);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (uint32_t i = 0; i < workers; i++) threads.emplace_back(worker_body);

    Progress progress("评测对局");
    progress.set_total(static_cast<int64_t>(config.games));
    if (config.progress_json) progress.set_json("evaluate", "games");
    uint64_t last_seen = 0;
    while (last_seen < config.games) {
        const uint64_t now = finished.load();
        if (now != last_seen) {
            last_seen = now;
            progress.update(static_cast<int64_t>(now));
        } else if (worker_error != nullptr) {
            break;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
    for (std::thread& thread : threads) thread.join();
    progress.finish();
    if (worker_error != nullptr) std::rethrow_exception(worker_error);

    EvaluateResult result;
    result.games = std::move(outcomes);
    result.seconds = monotonic_seconds() - start;

    std::vector<uint64_t> scores;
    scores.reserve(result.games.size());
    for (const GameOutcome& game : result.games) {
        scores.push_back(game.score);
        result.max_exponent = std::max(result.max_exponent, game.max_exponent);
    }
    if (!scores.empty()) {
        double sum = 0.0;
        for (const uint64_t score : scores) sum += static_cast<double>(score);
        result.mean_score = sum / static_cast<double>(scores.size());
        std::sort(scores.begin(), scores.end());
        const size_t middle = scores.size() / 2;
        result.median_score = scores.size() % 2 == 0
                                  ? 0.5 * static_cast<double>(scores[middle - 1] + scores[middle])
                                  : static_cast<double>(scores[middle]);
    }
    // 达标计数按“至少达到该块”统计：只有 2048 及以上的块有意义（游戏在 2048 之后继续）。
    for (int exponent = 11; exponent <= result.max_exponent; exponent++) {
        uint32_t count = 0;
        for (const GameOutcome& game : result.games) count += game.max_exponent >= exponent ? 1 : 0;
        result.tile_counts[exponent] = count;
    }

    if (!config.jsonl_path.empty()) {
        // 纯 C++ 轨道的协议变量只有这几项，其余全部写 null（“该策略没有这个变量”）。
        // 依据：本路径是整数运算，workers 实测逐字节一致（只是性能参数），没有浮点、
        // 没有批推理缓存，所以 lanes / in_flight / max_batch / cache_entries / device
        // 在这里不是“没记录”，而是根本不存在。清单与校验在 python 侧的 r2048.protocol，
        // 两边的字段必须一一对应——对不上时评分池会直接拒收这条记录，这是有意的。
        //
        // 没有单独入口的实现常量（rollout 探索常数 `RolloutMctsConfig::exploration`、
        // rollout 策略本身）挂在 protocol_version 上：改它们就要抬高该版本号。
        const bool search = config.policy == "mcts";
        ensure_parent_dir(config.jsonl_path);
        std::string text;
        for (const GameOutcome& game : result.games) {
            Json record = Json::object();
            record.set("episode", Json::integer(game.episode));
            record.set("label", Json::str(config.label.empty() ? config.policy : config.label));
            record.set("policy", Json::str(config.policy));
            record.set("seed", Json::integer(static_cast<int64_t>(config.seed)));
            record.set("score", Json::integer(static_cast<int64_t>(game.score)));
            record.set("max_exponent", Json::integer(game.max_exponent));
            record.set("steps", Json::integer(game.steps));
            record.set("track", Json::str(config.policy));
            record.set("rules_version", Json::integer(kRulesVersion));
            record.set("protocol_version", Json::integer(kProtocolVersion));
            record.set("simulations", search ? Json::integer(config.simulations) : Json());
            record.set("temperature_moves",
                       search ? Json::integer(config.temperature_moves) : Json());
            record.set("c_puct", Json());
            record.set("temperature", Json());
            record.set("inference_precision", Json());
            record.set("lanes", Json());
            record.set("in_flight", Json());
            record.set("max_batch", Json());
            record.set("cache_entries", Json());
            record.set("device", Json());
            text += record.dump(0);
            text += "\n";
        }
        write_text_atomic(config.jsonl_path, text);
    }
    return result;
}

}  // namespace r2048
