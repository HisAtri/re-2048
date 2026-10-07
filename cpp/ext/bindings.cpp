#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>

#include "core/version.h"
#include "game/board.h"
#include "game/rules.h"
#include "r2048_build_id.h"  // 由 CMake 生成，见 cmake/R2048BuildId.cmake
#include "search/batched_engine.h"

namespace py = pybind11;

namespace {

using U8Array = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using F32Array = py::array_t<float, py::array::c_style | py::array::forcecast>;
using U64Array = py::array_t<uint64_t, py::array::c_style | py::array::forcecast>;

r2048::Board board_from_array(const U8Array& array) {
    if (array.ndim() != 1 || array.shape(0) != r2048::kBoardCells) {
        throw std::runtime_error("棋盘必须是长度为 16 的 uint8 一维数组");
    }
    r2048::Board board;
    const uint8_t* data = array.data();
    for (int i = 0; i < r2048::kBoardCells; i++) board.tiles[i] = data[i];
    return board;
}

py::array_t<uint8_t> board_to_array(const r2048::Board& board) {
    py::array_t<uint8_t> out(r2048::kBoardCells);
    uint8_t* data = out.mutable_data();
    for (int i = 0; i < r2048::kBoardCells; i++) data[i] = board.tiles[i];
    return out;
}

r2048::EngineConfig engine_config_from_dict(const py::dict& dict) {
    r2048::EngineConfig config;
    const auto get_uint = [&](const char* key, uint32_t fallback) -> uint32_t {
        return dict.contains(key) ? dict[key].cast<uint32_t>() : fallback;
    };
    const auto get_uint64 = [&](const char* key, uint64_t fallback) -> uint64_t {
        return dict.contains(key) ? dict[key].cast<uint64_t>() : fallback;
    };
    const auto get_float = [&](const char* key, float fallback) -> float {
        return dict.contains(key) ? dict[key].cast<float>() : fallback;
    };
    config.lanes = get_uint("lanes", config.lanes);
    config.simulations = get_uint("simulations", config.simulations);
    config.in_flight_per_lane = get_uint("in_flight_per_lane", config.in_flight_per_lane);
    config.c_puct = get_float("c_puct", config.c_puct);
    config.temperature_moves = get_uint("temperature_moves", config.temperature_moves);
    config.temperature = get_float("temperature", config.temperature);
    config.cache_entries = get_uint("cache_entries", config.cache_entries);
    config.score_scale = get_float("score_scale", config.score_scale);
    config.seed = get_uint64("seed", config.seed);
    config.decision_mode = static_cast<r2048::DecisionMode>(get_uint("decision_mode", 0));
    config.collect_samples = get_uint("collect_samples", 1) != 0;
    config.episodes_target = get_uint("episodes_target", 0);
    config.resume = get_uint("resume", 0) != 0;
    if (dict.contains("control_path")) config.control_path = dict["control_path"].cast<std::string>();
    // 产出环境参数（device / max_batch）走"原样透传"：C++ 不认识它们的含义，
    // 但它们是数据来源记录的一部分，也让续跑时的参数闸门能发现"换了设备"。
    if (dict.contains("provenance")) {
        for (const auto& item : dict["provenance"].cast<py::dict>()) {
            config.provenance.emplace_back(item.first.cast<std::string>(),
                                           item.second.cast<std::string>());
        }
    }
    return config;
}

}  // namespace

PYBIND11_MODULE(r2048_core, m) {
    m.doc() = "2048 规则与搜索的 C++ 内核（Python 侧只负责神经网络与编排）";

    // 版本常量必须暴露给 Python：数据与检查点靠它们判断“这份数据是哪套语义产出的”。
    m.attr("RULES_VERSION") = r2048::kRulesVersion;
    m.attr("PROTOCOL_VERSION") = r2048::kProtocolVersion;
    m.attr("NUM_ACTIONS") = r2048::kNumActions;
    m.attr("SCORE_SCALE") = r2048::kScoreScale;
    m.attr("BUILD_ID") = std::string(R2048_BUILD_ID);
    // 动作顺序必须显式转成 int：C++ 的 unscoped enum 不会被 pybind11 自动转换。
    m.attr("UP") = static_cast<int>(r2048::kUp);
    m.attr("RIGHT") = static_cast<int>(r2048::kRight);
    m.attr("DOWN") = static_cast<int>(r2048::kDown);
    m.attr("LEFT") = static_cast<int>(r2048::kLeft);
    m.attr("REQUEST_VALUE") = static_cast<int>(r2048::kRequestValue);
    m.attr("REQUEST_POLICY") = static_cast<int>(r2048::kRequestPolicy);
    // 每步怎么选方向：搜索 / 值贪心 / 策略贪心。评测的三条 track 就靠它区分，
    // 而规则与对局推进是同一条代码路径，避免三条 track 用不同实现对局逻辑。
    m.attr("DECISION_SEARCH") = static_cast<int>(r2048::kDecisionSearch);
    m.attr("DECISION_VALUE_GREEDY") = static_cast<int>(r2048::kDecisionValueGreedy);
    m.attr("DECISION_POLICY_GREEDY") = static_cast<int>(r2048::kDecisionPolicyGreedy);

    // 规则函数：服务跨语言一致性测试与小规模调试。批量自对弈走下面的两阶段接口，
    // 不会逐样本跨语言调用。
    m.def(
        "apply_move",
        [](const U8Array& board, int action) {
            const r2048::MoveOutcome outcome = r2048::apply_move(board_from_array(board), action);
            py::dict result;
            result["board"] = board_to_array(outcome.board);
            result["score"] = outcome.score;
            result["valid"] = outcome.valid;
            return result;
        },
        py::arg("board"), py::arg("action"));

    m.def("legal_mask",
          [](const U8Array& board) { return r2048::legal_mask(board_from_array(board)); });

    py::class_<r2048::BatchedEngine> engine(m, "BatchedEngine");
    engine.doc() = R"(
批量 NN-MCTS 自对弈引擎。

协议是两阶段的：
    advance(max_requests) → 一批 (request_ids, request_kinds, boards)
    Python 一次前向 → submit(ids, values, policy_logits)
C++ 侧绝不逐样本调用 Python：一次决策节点展开会同时需要 1 个策略头结果（决策局面）
与最多 4 个价值头结果（各 afterstate），所以每个前向批次天然覆盖 5 倍以上的局面。

values 是**归一化**价值（分数 / score_scale），C++ 侧换算回原始分数。
)";
    engine.def(py::init([](const py::dict& config, const std::string& data_dir,
                           const std::string& batch_id, const std::string& model_id,
                           uint64_t samples_target, uint64_t shard_samples) {
                   return std::make_unique<r2048::BatchedEngine>(
                       engine_config_from_dict(config), data_dir, batch_id, model_id, samples_target,
                       shard_samples);
               }),
               py::arg("config"), py::arg("data_dir"), py::arg("batch_id"), py::arg("model_id"),
               py::arg("samples_target"), py::arg("shard_samples"));

    engine.def(
        "advance",
        [](r2048::BatchedEngine& self, uint32_t max_requests) {
            r2048::BatchedEngine::Advance advance;
            {
                // 树推进是纯 C++ 计算，期间释放 GIL：多线程驱动时不会互相卡住，
                // 也让 Python 侧可以在这段时间里继续做别的事情（例如另一批推理）。
                py::gil_scoped_release release;
                advance = self.advance(max_requests);
            }
            py::dict result;
            result["request_ids"] =
                py::array_t<uint64_t>(static_cast<py::ssize_t>(advance.request_ids.size()),
                                      advance.request_ids.data());
            result["request_kinds"] =
                py::array_t<uint8_t>(static_cast<py::ssize_t>(advance.request_kinds.size()),
                                     advance.request_kinds.data());
            const py::ssize_t rows = static_cast<py::ssize_t>(advance.request_ids.size());
            result["boards"] = py::array_t<uint8_t>({rows, static_cast<py::ssize_t>(r2048::kBoardCells)},
                                                    advance.boards.data());
            // 已结束的局用并列数组回传：一次评测可能结束上千局，逐局构造 Python 对象
            // 会把时间花在解释器上。
            const size_t episode_count = advance.episodes.size();
            std::vector<uint32_t> episode_ids(episode_count);
            std::vector<uint32_t> episode_steps(episode_count);
            std::vector<uint32_t> episode_tiles(episode_count);
            std::vector<uint64_t> episode_scores(episode_count);
            for (size_t i = 0; i < episode_count; i++) {
                episode_ids[i] = advance.episodes[i].episode;
                episode_steps[i] = advance.episodes[i].steps;
                episode_tiles[i] = advance.episodes[i].max_exponent;
                episode_scores[i] = advance.episodes[i].score;
            }
            result["episode_ids"] = py::array_t<uint32_t>(static_cast<py::ssize_t>(episode_count),
                                                         episode_ids.data());
            result["episode_steps"] = py::array_t<uint32_t>(static_cast<py::ssize_t>(episode_count),
                                                           episode_steps.data());
            result["episode_tiles"] = py::array_t<uint32_t>(static_cast<py::ssize_t>(episode_count),
                                                           episode_tiles.data());
            result["episode_scores"] = py::array_t<uint64_t>(static_cast<py::ssize_t>(episode_count),
                                                            episode_scores.data());
            result["done"] = advance.done;
            result["samples_written"] = advance.samples_written;
            result["episodes_written"] = advance.episodes_written;
            return result;
        },
        py::arg("max_requests"));

    engine.def(
        "abandon",
        [](r2048::BatchedEngine& self) {
            py::gil_scoped_release release;
            self.abandon();
        },
        "把当前在途请求全部重新排队（调用方丢弃了一批前向时用）。")
        .def(
        "submit",
        [](r2048::BatchedEngine& self, const U64Array& ids, const F32Array& values,
           const F32Array& policy_logits) {
            const size_t count = static_cast<size_t>(ids.size());
            if (values.size() != static_cast<py::ssize_t>(count)) {
                throw std::runtime_error("values 的长度必须与 request_ids 相同");
            }
            if (policy_logits.ndim() != 2 || static_cast<size_t>(policy_logits.shape(0)) != count ||
                policy_logits.shape(1) != r2048::kNumActions) {
                throw std::runtime_error("policy_logits 的形状必须是 (N, 4)");
            }
            {
                py::gil_scoped_release release;
                self.submit(ids.data(), values.data(), policy_logits.data(), count);
            }
        },
        py::arg("request_ids"), py::arg("values"), py::arg("policy_logits"));

    engine.def("stats", [](const r2048::BatchedEngine& self) {
        const r2048::EngineStats stats = self.stats();
        py::dict result;
        result["requests_issued"] = stats.requests_issued;
        result["requests_submitted"] = stats.requests_submitted;
        result["leaf_evaluations"] = stats.leaf_evaluations;
        result["leaf_from_cache"] = stats.leaf_from_cache;
        result["leaf_cache_rate"] = stats.leaf_cache_rate();
        result["simulations"] = stats.simulations;
        result["episodes"] = stats.episodes;
        result["samples"] = stats.samples;
        return result;
    });

    engine.def("debug_state", [](const r2048::BatchedEngine& self) {
        const r2048::BatchedEngine::DebugState state = self.debug_state();
        py::dict result;
        py::list lanes;
        for (const r2048::BatchedEngine::LaneStatus& lane : state.lanes) {
            py::dict entry;
            entry["episode"] = lane.episode;
            entry["step"] = lane.step;
            entry["simulations_done"] = lane.simulations_done;
            entry["pending"] = lane.pending;
            entry["outstanding"] = lane.outstanding;
            entry["finished"] = lane.finished;
            entry["needs_new_episode"] = lane.needs_new_episode;
            py::list missing;
            for (const r2048::BatchedEngine::MissingNeed& need : lane.missing) {
                py::dict item;
                item["leaf"] = need.leaf;
                item["kind"] = need.kind;
                item["missing_policy"] = need.missing_policy;
                item["missing_values"] = need.missing_values;
                missing.append(item);
            }
            entry["missing"] = missing;
            lanes.append(entry);
        }
        result["lanes"] = lanes;
        result["queued_requests"] = state.queued_requests;
        result["in_flight_requests"] = state.in_flight_requests;
        result["next_episode"] = state.next_episode;
        result["next_episode_to_write"] = state.next_episode_to_write;
        result["completed_buffered"] = state.completed_buffered;
        py::list keys;
        for (uint32_t key : state.completed_keys) keys.append(key);
        result["completed_keys"] = keys;
        result["samples_produced"] = state.samples_produced;
        result["samples_written"] = state.samples_written;
        result["reissues"] = state.reissues;
        result["stop_all"] = state.stop_all;
        return result;
    });

    engine.def(
        "request_stop",
        [](r2048::BatchedEngine& self) { self.request_stop(); },
        "请求在下一个合法停车点（局与局之间）停下并收尾：正在进行的局一定走完。\n"
        "Python 侧收到 KeyboardInterrupt 时走这条路，尾部数据因此不会丢。");
    engine.def("stop_requested", [](const r2048::BatchedEngine& self) {
        return self.stop_requested();
    });

    engine.def(
        "finish",
        [](r2048::BatchedEngine& self) {
            r2048::FinishResult result;
            {
                py::gil_scoped_release release;
                result = self.finish();
            }
            py::dict out;
            out["manifest_path"] = result.manifest_path;
            out["state"] = r2048::batch_state_name(result.state);
            out["samples"] = result.samples;
            out["episodes"] = result.episodes;
            return out;
        },
        "收尾：写完最后一个分片并落 manifest，返回 {manifest_path, state, samples, episodes}。\n"
        "state 为 complete 表示跑满目标，paused/stopped 表示被叫停（可以 --continue 接着跑）。");
}
