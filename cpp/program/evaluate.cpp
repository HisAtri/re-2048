#include "program/evaluate.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "core/fileio.h"
#include "core/json.h"
#include "core/logging.h"
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
        ensure_parent_dir(config.jsonl_path);
        std::string text;
        for (const GameOutcome& game : result.games) {
            Json record = Json::object();
            record.set("episode", Json::integer(game.episode));
            record.set("label", Json::str(config.label.empty() ? config.policy : config.label));
            record.set("policy", Json::str(config.policy));
            record.set("simulations", Json::integer(config.simulations));
            record.set("seed", Json::integer(static_cast<int64_t>(config.seed)));
            record.set("score", Json::integer(static_cast<int64_t>(game.score)));
            record.set("max_exponent", Json::integer(game.max_exponent));
            record.set("steps", Json::integer(game.steps));
            text += record.dump(0);
            text += "\n";
        }
        write_text_atomic(config.jsonl_path, text);
    }
    return result;
}

}  // namespace r2048
