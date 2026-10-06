#include "search/batched_engine.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <map>
#include <unordered_map>
#include <mutex>
#include <stdexcept>

#include "core/fileio.h"
#include "core/logging.h"
#include "dataio/sample.h"
#include "dataio/shard.h"
#include "game/rules.h"
#include "game/seeds.h"
#include "program/episode.h"  // opening_board：开局口径必须与纯 C++ 跑谱完全一致

namespace r2048 {

struct BatchedEngine::Impl {
    static constexpr uint32_t kNoNode = 0xFFFFFFFFu;
    static constexpr int kMaxChanceChildren = 32;
    static constexpr int kMaxPumpRounds = 128;
    static constexpr size_t kProbeLimit = 16;  // 线性探测上限：定长表的插入/查找都是 O(1)

    enum NodeKind : uint8_t { kDecision = 0, kChance = 1 };
    enum NodeState : uint8_t { kFresh = 0, kWaiting = 1, kEvaluated = 2 };

    struct Node {
        Board board;
        uint8_t kind = kDecision;
        uint8_t state = kFresh;
        uint8_t legal_mask = 0;
        uint8_t parent_action = 0;
        uint8_t chance_children = 0;
        uint32_t parent = kNoNode;
        uint32_t visits = 0;
        float value_sum = 0.0f;
        float prior[kNumActions] = {0, 0, 0, 0};
        float one_ply[kNumActions] = {0, 0, 0, 0};  // r(s,a) + A(u_a)：未访问边的 Q 初值
        float edge_q_sum[kNumActions] = {0, 0, 0, 0};
        uint32_t edge_visits[kNumActions] = {0, 0, 0, 0};
        float edge_reward[kNumActions] = {0, 0, 0, 0};
        uint32_t edge_child[kNumActions] = {kNoNode, kNoNode, kNoNode, kNoNode};
        Board edge_afterstate[kNumActions];
        float afterstate_value = 0.0f;  // 随机块节点：A(u)（原始分数）
        uint32_t chance_keys[kMaxChanceChildren] = {};
        uint32_t chance_child[kMaxChanceChildren] = {};
    };

    // 一条已降到底、等待神经网络求值的模拟。path 是它走过的节点序列（用于回传），
    // leaf 是要被求值的节点。在途多条模拟是吞吐的前提，见 in_flight_per_lane。
    struct Pending {
        std::vector<uint32_t> path;
        uint32_t leaf = 0;
    };

    struct Lane {
        uint32_t episode = 0;
        Board board;
        uint8_t legal_mask = 0;
        uint32_t step = 0;
        uint64_t total_score = 0;
        int max_exponent = 0;
        bool finished = false;           // 本 lane 已停止工作
        bool needs_new_episode = false;  // 上一局已结束，等待开新局
        std::vector<Node> nodes;
        uint32_t simulations_done = 0;
        std::vector<Pending> pending;
        uint32_t outstanding = 0;  // 已发出、尚未回填的请求条数
        EpisodeRecorder recorder{0, 0};
    };

    struct Request {
        uint64_t id;
        uint8_t kind;
        uint8_t board[kBoardCells];
        uint32_t lane;
    };

    Impl(const EngineConfig& config_in, std::string data_dir_in, std::string batch_id_in,
         std::string model_id_in, uint64_t samples_target_in, uint64_t shard_samples_in)
        : config(config_in),
          data_dir(std::move(data_dir_in)),
          batch_id(std::move(batch_id_in)),
          model_id(std::move(model_id_in)),
          samples_target(samples_target_in) {
        if (config.lanes == 0) throw std::runtime_error("lane 数必须大于 0");
        if (config.simulations == 0) throw std::runtime_error("模拟次数必须大于 0");
        if (config.simulations > 0xFFFF) throw std::runtime_error("模拟次数上限为 65535");
        if (config.collect_samples) {
            if (!dir_is_usable_as_output(data_dir)) {
                throw std::runtime_error("输出目录已存在且非空，拒绝覆盖: " + data_dir);
            }
            ensure_dir(data_dir);
            writer = std::make_unique<ShardWriter>(data_dir, shard_samples_in);
        }
        // 表长取 2 的幂，容量至少是配置值的一倍（保持低负载因子）。
        size_t capacity = 64;
        while (capacity < static_cast<size_t>(config.cache_entries)) capacity <<= 1;
        cache_mask = capacity - 1;
        cache.assign(capacity, CacheEntry{});
        lanes.resize(config.lanes);
        for (uint32_t i = 0; i < config.lanes; i++) start_episode(lanes[i]);
    }

    // ---------------- 推理缓存 ----------------
    // 调用约定：cache_get / cache_put 都必须在持有 cache_mutex 时调用。
    //
    // 结构：定长开放寻址表（线性探测，插入上限 kProbeLimit 个槽）。
    // 为什么不是“2 路组相联数组”：那种结构下只要有几个高频局面落在同一个 set 里，
    // 它们的条目就会互相驱逐，而一个决策节点需要“策略 + 最多 4 个 afterstate 价值”
    // 同时命中才能展开——任何一个被挤掉就永远无法展开，表现为同一批请求被无限补发。
    // 这是实测踩到过的活锁（当时 2577 个不同局面里有 3 个高频局面撞进同一个 set）。
    // 开放寻址表容量远大于工作集（默认 2^17 槽 vs 数百个在途局面），几乎不会发生驱逐。
    //
    // 不变式：槽位只会从“空”变成“占用”，从不被清空。线性探测的查找因此可以在
    // 遇到第一个空槽时安全终止；驱逐时直接覆盖探到的槽，不制造空洞。

    struct CacheEntry {
        uint8_t board[kBoardCells] = {};
        uint8_t kind = 0;
        uint8_t valid = 0;
        float value = 0.0f;
        float logits[kNumActions] = {0, 0, 0, 0};
    };

    size_t cache_slot_of(const uint8_t* board, uint8_t kind) const {
        const uint64_t mixed =
            board_hash(board) ^ (static_cast<uint64_t>(kind) * 0x9e3779b97f4a7c15ull);
        // 用高位做索引：乘法型混合器的高位熵更好。
        return static_cast<size_t>(mixed >> 32) & cache_mask;
    }

    // 本轮刚回填的答案放在一张独立的小表里，它在整个 advance 期间**不会被驱逐**。
    //
    // 为什么必须有这一层：一个决策节点要同时命中“策略 + 最多 4 个 afterstate 价值”才能
    // 展开。如果这些结果只能躺在会驱逐的缓存里，某些条目可能在解析前就被挤掉，于是
    // 同一批请求被无限补发（实测踩到过：2577 个不同局面里几个高频局面反复互相驱逐）。
    // 有了这层之后，只要一个挂起模拟缺的几条结果在同一批里回填，它就一定能被解析，
    // 进度不再依赖缓存容量，缓存只负责减少跨语言往返。
    struct AnswerKey {
        uint8_t board[kBoardCells];
        uint8_t kind;
        bool operator==(const AnswerKey& other) const {
            return kind == other.kind && std::memcmp(board, other.board, kBoardCells) == 0;
        }
    };
    struct AnswerKeyHash {
        size_t operator()(const AnswerKey& key) const { return static_cast<size_t>(board_hash(key.board)); }
    };
    struct AnswerValue {
        float value = 0.0f;
        float logits[kNumActions] = {0, 0, 0, 0};
    };

    bool fresh_get(const Board& board, uint8_t kind, float* value, float* logits) {
        AnswerKey key;
        std::memcpy(key.board, board.tiles, kBoardCells);
        key.kind = kind;
        const auto found = fresh_answers.find(key);
        if (found == fresh_answers.end()) return false;
        if (value != nullptr) *value = found->second.value;
        if (logits != nullptr) {
            for (int action = 0; action < kNumActions; action++) logits[action] = found->second.logits[action];
        }
        return true;
    }

    void fresh_put(const Board& board, uint8_t kind, float value, const float* logits) {
        AnswerKey key;
        std::memcpy(key.board, board.tiles, kBoardCells);
        key.kind = kind;
        AnswerValue answer;
        answer.value = value;
        if (logits != nullptr) {
            for (int action = 0; action < kNumActions; action++) answer.logits[action] = logits[action];
        }
        fresh_answers[key] = answer;
    }

    bool cache_get(const Board& board, uint8_t kind, float* value, float* logits) {
        if (fresh_get(board, kind, value, logits)) return true;
        const size_t base = cache_slot_of(board.tiles, kind);
        for (size_t probe = 0; probe < kProbeLimit; probe++) {
            CacheEntry& entry = cache[(base + probe) & cache_mask];
            if (entry.valid == 0) return false;  // 空槽即可断定不存在
            if (entry.kind != kind) continue;
            if (std::memcmp(entry.board, board.tiles, kBoardCells) != 0) continue;
            if (value != nullptr) *value = entry.value;
            if (logits != nullptr) {
                for (int action = 0; action < kNumActions; action++) logits[action] = entry.logits[action];
            }
            return true;
        }
        return false;
    }

    void cache_put(const Board& board, uint8_t kind, float value, const float* logits) {
        fresh_put(board, kind, value, logits);
        const size_t base = cache_slot_of(board.tiles, kind);
        size_t target = base;
        for (size_t probe = 0; probe < kProbeLimit; probe++) {
            const size_t slot = (base + probe) & cache_mask;
            if (cache[slot].valid == 0) {
                target = slot;
                break;
            }
            if (cache[slot].kind == kind && std::memcmp(cache[slot].board, board.tiles, kBoardCells) == 0) {
                target = slot;  // 已存在：原地更新
                break;
            }
            target = slot;  // 探测范围内全满时，覆盖第一个探到的槽
        }
        CacheEntry& entry = cache[target];
        std::memcpy(entry.board, board.tiles, kBoardCells);
        entry.kind = kind;
        entry.valid = 1;
        entry.value = value;
        if (logits != nullptr) {
            for (int action = 0; action < kNumActions; action++) entry.logits[action] = logits[action];
        }
    }

    // ---------------- 对局与搜索树 ----------------

    uint32_t add_node(Lane& lane, const Board& board, NodeKind kind, uint32_t parent,
                      uint8_t parent_action) {
        Node node;
        node.board = board;
        node.kind = static_cast<uint8_t>(kind);
        node.parent = parent;
        node.parent_action = parent_action;
        if (kind == kDecision) {
            node.legal_mask = legal_mask(board);
            for (int action = 0; action < kNumActions; action++) {
                if ((node.legal_mask & (1u << action)) == 0) continue;
                const MoveOutcome outcome = apply_move(board, action);
                node.edge_afterstate[action] = outcome.board;
                node.edge_reward[action] = static_cast<float>(outcome.score);
            }
            // 终局节点没有可求值的内容，直接视为已求值。
            if (node.legal_mask == 0) node.state = kEvaluated;
        }
        lane.nodes.push_back(node);
        return static_cast<uint32_t>(lane.nodes.size() - 1);
    }

    // 贪心模式不做树搜索，只需要把根局面求值一次（拿到先验与各 afterstate 的价值）。
    uint32_t effective_simulations() const {
        return config.decision_mode == kDecisionSearch ? config.simulations : 1;
    }

    void start_episode(Lane& lane) {
        // 必须是原子取号：lane 是并行推进的，普通自增会丢更新，
        // 于是局号出现空洞，而写入侧按局号顺序落盘，会永远等一个不会再出现的局号。
        lane.episode = next_episode.fetch_add(1);
        lane.board = opening_board(config.seed, lane.episode);
        lane.legal_mask = legal_mask(lane.board);
        lane.step = 0;
        lane.total_score = 0;
        lane.max_exponent = lane.board.max_exponent();
        lane.finished = false;
        lane.needs_new_episode = false;
        lane.simulations_done = 0;
        lane.pending.clear();
        lane.recorder = EpisodeRecorder(lane.episode, static_cast<uint16_t>(config.simulations));
        reset_tree(lane);
    }

    void reset_tree(Lane& lane) {
        lane.nodes.clear();
        lane.pending.clear();
        lane.simulations_done = 0;
        lane.nodes.reserve(static_cast<size_t>(effective_simulations()) * 4 + 8);
        add_node(lane, lane.board, kDecision, kNoNode, 0);
    }

    // 叶子求值所需的结果是否都在缓存里（终局节点天然就绪）。
    bool needs_satisfied(const Lane& lane, uint32_t node_index) {
        const Node& node = lane.nodes[node_index];
        if (node.kind == kDecision) {
            if (node.legal_mask == 0) return true;
            if (!cache_get(node.board, kRequestPolicy, nullptr, nullptr)) return false;
            for (int action = 0; action < kNumActions; action++) {
                if ((node.legal_mask & (1u << action)) == 0) continue;
                if (!cache_get(node.edge_afterstate[action], kRequestValue, nullptr, nullptr)) return false;
            }
            return true;
        }
        return cache_get(node.board, kRequestValue, nullptr, nullptr);
    }



    // 发出这个叶子缺失的请求。
    //
    // 锁纪律：cache_mutex 与 bookkeeping_mutex **永不嵌套**。两把锁分别在不同阶段获取，
    // 否则 pump（cache → bookkeeping）与 submit（bookkeeping → cache）会构成死锁环，
    // 表现为整个引擎随机卡死。
    void issue_requests(Lane& lane, uint32_t lane_index, uint32_t node_index) {
        const uint8_t kind = lane.nodes[node_index].kind;
        const uint8_t mask = lane.nodes[node_index].legal_mask;

        // 关键：**要么不请求，要么把该节点需要的全部结果一起请求**（策略 + 所有合法 afterstate
        // 的价值），而不是只请求当下缺的那几条。
        //
        // 只请求缺失项会造成活锁：一个决策节点要同时拿到 5 条结果才能展开，若其中一条是
        // 上一轮取回、随后被缓存驱逐的，那么“补缺失项”会永远只补到一部分——补回来的又把
        // 别的挤掉，轮轮如此。整组一起请求之后，同一条挂起模拟需要的所有结果都落在同一个
        // 批次里，紧接着的回填就一定能让它就绪（配合 submit 里逐条回填即解析）。
        std::vector<Board> fresh_boards;
        std::vector<uint8_t> fresh_kinds;
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            const Board board = lane.nodes[node_index].board;
            if (kind == kDecision) {
                fresh_boards.push_back(board);
                fresh_kinds.push_back(kRequestPolicy);
                for (int action = 0; action < kNumActions; action++) {
                    if ((mask & (1u << action)) == 0) continue;
                    fresh_boards.push_back(lane.nodes[node_index].edge_afterstate[action]);
                    fresh_kinds.push_back(kRequestValue);
                }
            } else {
                fresh_boards.push_back(board);
                fresh_kinds.push_back(kRequestValue);
            }
        }

        if (!fresh_boards.empty()) {
            std::lock_guard<std::mutex> lock(bookkeeping_mutex);
            for (size_t i = 0; i < fresh_boards.size(); i++) {
                Request request;
                request.id = next_request_id++;
                request.kind = fresh_kinds[i];
                std::memcpy(request.board, fresh_boards[i].tiles, kBoardCells);
                request.lane = lane_index;
                requests.push_back(request);
            }
        }
        stat_requests.fetch_add(fresh_boards.size());
        lane.outstanding += static_cast<uint32_t>(fresh_boards.size());
        lane.nodes[node_index].state = kWaiting;
    }

    float evaluate_node(Lane& lane, uint32_t node_index) {
        Node& node = lane.nodes[node_index];
        if (node.kind == kChance) {
            float value = 0.0f;
            if (!cache_get(node.board, kRequestValue, &value, nullptr)) {
                throw std::runtime_error("随机块节点的价值不在缓存里");
            }
            node.afterstate_value = value * config.score_scale;
            node.state = kEvaluated;
            return node.afterstate_value;
        }
        if (node.legal_mask == 0) {
            node.state = kEvaluated;
            return 0.0f;
        }

        float logits[kNumActions] = {0, 0, 0, 0};
        if (!cache_get(node.board, kRequestPolicy, nullptr, logits)) {
            throw std::runtime_error("决策节点的策略结果不在缓存里");
        }
        float max_logit = -FLT_MAX;
        for (int action = 0; action < kNumActions; action++) {
            if ((node.legal_mask & (1u << action)) == 0) continue;
            max_logit = std::max(max_logit, logits[action]);
        }
        float sum = 0.0f;
        for (int action = 0; action < kNumActions; action++) {
            if ((node.legal_mask & (1u << action)) == 0) {
                node.prior[action] = 0.0f;
                continue;
            }
            node.prior[action] = std::exp(logits[action] - max_logit);
            sum += node.prior[action];
        }
        float best = -FLT_MAX;
        for (int action = 0; action < kNumActions; action++) {
            if ((node.legal_mask & (1u << action)) == 0) continue;
            node.prior[action] = sum > 0.0f ? node.prior[action] / sum : 1.0f;
            float afterstate_value = 0.0f;
            if (!cache_get(node.edge_afterstate[action], kRequestValue, &afterstate_value, nullptr)) {
                throw std::runtime_error("afterstate 的价值不在缓存里");
            }
            node.one_ply[action] = node.edge_reward[action] + afterstate_value * config.score_scale;
            best = std::max(best, node.one_ply[action]);
        }
        node.state = kEvaluated;
        return best;
    }

    int select_puct(const Node& node) const {
        float min_q = FLT_MAX;
        float max_q = -FLT_MAX;
        for (int action = 0; action < kNumActions; action++) {
            if ((node.legal_mask & (1u << action)) == 0) continue;
            const float q = node.edge_visits[action] > 0
                                ? node.edge_q_sum[action] / static_cast<float>(node.edge_visits[action])
                                : node.one_ply[action];
            min_q = std::min(min_q, q);
            max_q = std::max(max_q, q);
        }
        const float range = max_q - min_q;
        const float sqrt_visits = std::sqrt(static_cast<float>(node.visits > 0 ? node.visits : 1));

        int best_action = -1;
        float best_score = -FLT_MAX;
        for (int action = 0; action < kNumActions; action++) {
            if ((node.legal_mask & (1u << action)) == 0) continue;
            const float q = node.edge_visits[action] > 0
                                ? node.edge_q_sum[action] / static_cast<float>(node.edge_visits[action])
                                : node.one_ply[action];
            const float normalized = range > 1e-6f ? (q - min_q) / range : 0.0f;
            const float exploration = config.c_puct * node.prior[action] * sqrt_visits /
                                      (1.0f + static_cast<float>(node.edge_visits[action]));
            const float score = normalized + exploration;
            if (score > best_score) {
                best_score = score;
                best_action = action;
            }
        }
        if (best_action < 0) throw std::runtime_error("决策节点没有合法方向");
        return best_action;
    }

    void backup(Lane& lane, const std::vector<uint32_t>& path, float leaf_value) {
        // 路径形如 [根决策, 随机块, 决策, 随机块, …, 叶子]。穿过“随机块 → 决策父节点”
        // 时把该边的立即得分计入返回值，于是父节点的 Q 自然等于 r(s,a) + A(u)。
        float value = leaf_value;
        for (size_t i = path.size(); i-- > 0;) {
            const uint32_t current = path[i];
            Node& node = lane.nodes[current];
            node.visits++;
            node.value_sum += value;
            if (node.parent == kNoNode) break;
            Node& parent = lane.nodes[node.parent];
            if (parent.kind == kDecision) {
                const int action = node.parent_action;
                parent.edge_q_sum[action] += parent.edge_reward[action] + value;
                value += parent.edge_reward[action];
            }
        }
    }

    uint32_t sample_chance_child(Lane& lane, uint32_t chance_index, Rng& rng) {
        const int empties = lane.nodes[chance_index].board.empty_count();
        if (empties == 0) throw std::runtime_error("随机块节点没有空格：afterstate 状态不合法");
        int cell_pick = static_cast<int>(rng.below(static_cast<uint32_t>(empties)));
        int cell = 0;
        for (; cell < kBoardCells; cell++) {
            if (lane.nodes[chance_index].board.tiles[cell] != 0) continue;
            if (cell_pick == 0) break;
            cell_pick--;
        }
        const bool is_four = rng.one_in_ten();
        const uint32_t key = static_cast<uint32_t>(cell) * 2u + (is_four ? 1u : 0u);
        const uint8_t children = lane.nodes[chance_index].chance_children;
        for (uint8_t i = 0; i < children; i++) {
            if (lane.nodes[chance_index].chance_keys[i] == key) {
                return lane.nodes[chance_index].chance_child[i];
            }
        }
        Board spawned = lane.nodes[chance_index].board;
        spawned.tiles[cell] = is_four ? 2 : 1;
        const uint32_t child = add_node(lane, spawned, kDecision, chance_index, 0);
        lane.nodes[chance_index].chance_keys[children] = key;
        lane.nodes[chance_index].chance_child[children] = child;
        lane.nodes[chance_index].chance_children = static_cast<uint8_t>(children + 1);
        return child;
    }

    // 一条模拟完成：求值叶子并回传。
    void finish_simulation(Lane& lane, const std::vector<uint32_t>& path, uint32_t leaf) {
        const float leaf_value = [&] {
            std::lock_guard<std::mutex> lock(cache_mutex);
            return evaluate_node(lane, leaf);
        }();
        backup(lane, path, leaf_value);
        lane.simulations_done++;
        stat_simulations.fetch_add(1);
    }

    // 解决所有“结果已就绪”的挂起模拟，返回解决条数。
    uint32_t resolve_ready(Lane& lane) {
        uint32_t resolved = 0;
        for (size_t index = 0; index < lane.pending.size();) {
            bool ready = false;
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                ready = needs_satisfied(lane, lane.pending[index].leaf);
            }
            if (!ready) {
                index++;
                continue;
            }
            const Pending pending = lane.pending[index];
            lane.pending.erase(lane.pending.begin() + static_cast<long>(index));
            finish_simulation(lane, pending.path, pending.leaf);
            resolved++;
        }
        return resolved;
    }

    // 从根降到一个待求值的叶子：能从缓存直接求值就完成这条模拟，否则挂起等待神经网络。
    void start_simulation(Lane& lane, uint32_t lane_index) {
        Pending pending;
        pending.path.push_back(0);
        // 采样种子带上“第几条在途模拟”，使同一 lane 的多次模拟不共享随机序列。
        Rng spawn_rng(derive_seed(config.seed, kSaltEnvironment, lane.episode, lane.step,
                                  1 + lane.simulations_done + lane.pending.size()));

        uint32_t node_index = 0;
        for (;;) {
            if (lane.nodes[node_index].state != kEvaluated) {
                // needs_satisfied 只读缓存；issue_requests 自己会分两段取锁。
                // 两者不能在同一个 cache_mutex 临界区里串起来，否则会自己锁死自己。
                bool satisfied = false;
                {
                    std::lock_guard<std::mutex> lock(cache_mutex);
                    satisfied = needs_satisfied(lane, node_index);
                }
                if (!satisfied) issue_requests(lane, lane_index, node_index);
                stat_leaf.fetch_add(1);
                if (!satisfied) {
                    pending.leaf = node_index;
                    lane.pending.push_back(std::move(pending));
                    return;
                }
                (void)0;
                stat_leaf_cached.fetch_add(1);
                finish_simulation(lane, pending.path, node_index);
                return;
            }

            if (lane.nodes[node_index].kind == kChance) {
                node_index = sample_chance_child(lane, node_index, spawn_rng);
                pending.path.push_back(node_index);
                continue;
            }
            if (lane.nodes[node_index].legal_mask == 0) {
                // 终局：价值为 0，无需神经网络。
                backup(lane, pending.path, 0.0f);
                lane.simulations_done++;
                stat_simulations.fetch_add(1);
                return;
            }
            const int action = select_puct(lane.nodes[node_index]);
            uint32_t child = lane.nodes[node_index].edge_child[action];
            if (child == kNoNode) {
                const Board afterstate = lane.nodes[node_index].edge_afterstate[action];
                child = add_node(lane, afterstate, kChance, node_index, static_cast<uint8_t>(action));
                lane.nodes[node_index].edge_child[action] = child;
            }
            lane.nodes[node_index].edge_visits[action]++;
            node_index = child;
            pending.path.push_back(node_index);
        }
    }

    void finish_step(Lane& lane) {
        const Node& root = lane.nodes[0];
        uint16_t visits[kNumActions] = {0, 0, 0, 0};
        uint32_t total_visits = 0;
        for (int action = 0; action < kNumActions; action++) {
            visits[action] = static_cast<uint16_t>(std::min<uint32_t>(root.edge_visits[action], 0xFFFF));
            total_visits += visits[action];
        }

        int action = -1;
        if (config.decision_mode == kDecisionValueGreedy) {
            // 纯价值贪心：不搜索，直接取 r(s,a) + A(u_a) 最大的方向。评测里这是
            // “只考察价值头”的那条 track；取最大而非采样，保证同局面可复现。
            float best = -FLT_MAX;
            for (int a = 0; a < kNumActions; a++) {
                if ((lane.legal_mask & (1u << a)) == 0) continue;
                if (root.one_ply[a] > best) {
                    best = root.one_ply[a];
                    action = a;
                }
            }
        } else if (config.decision_mode == kDecisionPolicyGreedy) {
            // 纯策略贪心：只考察策略头，同样不搜索。
            float best = -FLT_MAX;
            for (int a = 0; a < kNumActions; a++) {
                if ((lane.legal_mask & (1u << a)) == 0) continue;
                if (root.prior[a] > best) {
                    best = root.prior[a];
                    action = a;
                }
            }
        } else if (lane.step < config.temperature_moves && config.temperature > 0.0f && total_visits > 0) {
            // 开局按根访问分布采样：自对弈数据需要多样性。
            Rng rng(derive_seed(config.seed, kSaltAction, lane.episode, lane.step, 0));
            const uint32_t pick = rng.below(total_visits);
            uint32_t accum = 0;
            for (int a = 0; a < kNumActions; a++) {
                accum += visits[a];
                if (pick < accum) {
                    action = a;
                    break;
                }
            }
        } else {
            // 之后取访问数最大的方向；并列时取编号更小者，保证可复现。
            uint16_t best = 0;
            for (int a = 0; a < kNumActions; a++) {
                if (visits[a] > best) {
                    best = visits[a];
                    action = a;
                }
            }
        }
        if (action < 0) {
            // 模拟次数少于合法方向数时可能出现全零访问：退回编号最小的合法方向。
            for (int a = 0; a < kNumActions; a++) {
                if ((lane.legal_mask & (1u << a)) != 0) {
                    action = a;
                    break;
                }
            }
        }
        if (action < 0) throw std::runtime_error("无法为本步选出任何方向");

        const MoveOutcome move = apply_move(lane.board, action);
        if (config.collect_samples) {
            lane.recorder.add(lane.board, move.board, action, lane.legal_mask,
                              static_cast<uint32_t>(move.score), visits);
        }
        lane.total_score += move.score;
        const int exponent = move.board.max_exponent();
        lane.max_exponent = exponent > lane.max_exponent ? exponent : lane.max_exponent;
        lane.board = move.board;

        Rng rng(derive_seed(config.seed, kSaltEnvironment, lane.episode, lane.step + 1, 0));
        spawn_tile(lane.board, rng);

        lane.step++;
        lane.legal_mask = legal_mask(lane.board);
        if (lane.legal_mask == 0) {
            EpisodeResult outcome;
            outcome.episode = lane.episode;
            outcome.steps = lane.step;
            outcome.max_exponent = static_cast<uint32_t>(lane.max_exponent);
            outcome.score = lane.total_score;
            if (config.collect_samples) {
                std::vector<Sample> samples = lane.recorder.finish();
                std::lock_guard<std::mutex> lock(bookkeeping_mutex);
                samples_produced += samples.size();
                completed[lane.episode] = std::move(samples);
                if (config.episodes_target == 0 && samples_produced >= samples_target) stop_all.store(true);
                finished_outcomes.push_back(outcome);
            } else {
                std::lock_guard<std::mutex> lock(bookkeeping_mutex);
                finished_outcomes.push_back(outcome);
            }
            stat_episodes.fetch_add(1);
            // 只在这里（一局结束时）允许停下：正在进行的局必须走完，
            // 否则会留下“清单需要的局号永远不出现”的空洞，写入侧就会卡在半途。
            lane.needs_new_episode = true;
            return;
        }
        reset_tree(lane);
    }

    // 按局号顺序写盘：写入内容因此与 lane 调度顺序完全无关。
    void flush_completed() {
        while (true) {
            const auto found = completed.find(next_episode_to_write);
            if (found == completed.end()) break;
            std::vector<Sample> samples = std::move(found->second);
            completed.erase(found);

            if (written_samples() >= samples_target) {
                stop_all = true;
                break;
            }
            if (written_samples() + samples.size() > samples_target) {
                // 最后一个不完整的局被截断：样本数恰好等于目标条数，
                // 且截断只依赖目标，与线程数、调度顺序无关。
                samples.resize(static_cast<size_t>(samples_target - written_samples()));
                stop_all = true;
            }
            writer->add_episode(next_episode_to_write, samples);
            next_episode_to_write++;
            if (written_samples() >= samples_target) stop_all = true;
        }
    }

    void pump_all(uint32_t max_requests) {
        for (int round = 0; round < kMaxPumpRounds; round++) {
            bool progressed = false;
            for (uint32_t lane_index = 0; lane_index < lanes.size(); lane_index++) {
                Lane& lane = lanes[lane_index];
                if (lane.finished) continue;
                while (!lane.finished) {
                    if (lane.needs_new_episode) {
                        // 换局处才是合法的停车点：目标达成后不再开新局，
                        // 但已经开始的局一定走完。
                        if (stop_all.load()) {
                            lane.finished = true;
                            break;
                        }
                        if (config.episodes_target > 0 && next_episode.load() >= config.episodes_target) {
                            lane.finished = true;
                            break;
                        }
                        start_episode(lane);
                    }
                    // 先把结果已就绪的挂起模拟结算掉，再继续铺设新的模拟。
                    if (resolve_ready(lane) > 0) progressed = true;

                    if (lane.simulations_done >= effective_simulations()) {
                        finish_step(lane);
                        progressed = true;
                        continue;
                    }
                    if (lane.pending.size() >= config.in_flight_per_lane ||
                        lane.pending.size() >= effective_simulations() - lane.simulations_done) {
                        // 在途已满。这里必须处理一种会死锁的情形：缓存把某个挂起模拟
                        // 需要的结果挤掉了，于是它永远不会就绪，而 lane 又因为“在途已满”
                        // 不再开工。因此在没有任何在途请求时，为未就绪的挂起模拟补发请求。
                        if (lane.outstanding == 0) {
                            std::vector<uint32_t> missing_leaves;
                            {
                                std::lock_guard<std::mutex> lock(cache_mutex);
                                for (const Pending& pending : lane.pending) {
                                    if (!needs_satisfied(lane, pending.leaf)) {
                                        missing_leaves.push_back(pending.leaf);
                                    }
                                }
                            }
                            for (const uint32_t leaf : missing_leaves) {
                                issue_requests(lane, lane_index, leaf);
                            }
                            if (!missing_leaves.empty()) {
                                progressed = true;
                                stat_reissues.fetch_add(missing_leaves.size());
                            }
                        }
                        break;
                    }
                    start_simulation(lane, lane_index);
                    progressed = true;
                }
            }
            if (pending_request_count() >= max_requests) break;
            if (!progressed) break;
        }
    }

    bool all_lanes_finished() const {
        for (const Lane& lane : lanes) {
            if (!lane.finished) return false;
        }
        return true;
    }

    uint64_t written_samples() const { return writer ? writer->samples_written() : 0; }

    size_t pending_request_count() {
        std::lock_guard<std::mutex> lock(bookkeeping_mutex);
        return requests.size() + in_flight.size();
    }

    // 诊断：一条挂起模拟到底缺哪一项结果。卡住时这是唯一能直接指认原因的字段。
    void fill_missing(const Lane& lane, BatchedEngine::LaneStatus& status) {
        for (const Pending& pending : lane.pending) {
            const Node& node = lane.nodes[pending.leaf];
            BatchedEngine::MissingNeed need;
            need.leaf = pending.leaf;
            need.kind = node.kind;
            if (node.kind == kDecision) {
                need.missing_policy = !cache_get(node.board, kRequestPolicy, nullptr, nullptr);
                for (int action = 0; action < kNumActions; action++) {
                    if ((node.legal_mask & (1u << action)) == 0) continue;
                    if (!cache_get(node.edge_afterstate[action], kRequestValue, nullptr, nullptr)) {
                        need.missing_values++;
                    }
                }
            } else {
                need.missing_values = cache_get(node.board, kRequestValue, nullptr, nullptr) ? 0 : 1;
            }
            status.missing.push_back(need);
        }
    }

    // ---------------- 成员 ----------------

    EngineConfig config;
    std::string data_dir;
    std::string batch_id;
    std::string model_id;
    uint64_t samples_target;
    std::unique_ptr<ShardWriter> writer;
    std::vector<Lane> lanes;

    // 以下成员受 bookkeeping_mutex 保护（会被并行推进的 lane 访问）
    std::mutex bookkeeping_mutex;
    std::vector<Request> requests;                 // 已发出、尚未交给 Python
    std::map<uint64_t, Request> in_flight;         // 已交给 Python、等待 submit
    std::map<uint32_t, std::vector<Sample>> completed;
    std::vector<EpisodeResult> finished_outcomes;  // 待回传给 Python 的已结束局
    uint64_t next_request_id = 1;
    uint64_t samples_produced = 0;
    std::atomic<bool> stop_all{false};
    std::atomic<uint64_t> stat_requests{0};
    std::atomic<uint64_t> stat_submitted{0};
    std::atomic<uint64_t> stat_leaf{0};
    std::atomic<uint64_t> stat_leaf_cached{0};
    std::atomic<uint64_t> stat_simulations{0};
    std::atomic<uint64_t> stat_episodes{0};
    std::atomic<uint64_t> stat_reissues{0};

    // cache_mutex 保护缓存；cache 成员本身的长度在构造后不再变化。
    mutable std::mutex cache_mutex;
    size_t cache_mask = 0;
    std::vector<CacheEntry> cache;
    // 本轮回填的答案（每个 advance 开始时清空）。见 cache_get 里的说明。
    std::unordered_map<AnswerKey, AnswerValue, AnswerKeyHash> fresh_answers;

    EngineStats stats;  // 只在 finish() 里做一次快照
    std::atomic<uint32_t> next_episode{0};
    uint32_t next_episode_to_write = 0;
    bool finished = false;
};

BatchedEngine::BatchedEngine(const EngineConfig& config, std::string data_dir, std::string batch_id,
                             std::string model_id, uint64_t samples_target, uint64_t shard_samples)
    : impl_(std::make_unique<Impl>(config, std::move(data_dir), std::move(batch_id), std::move(model_id),
                                   samples_target, shard_samples)) {}

BatchedEngine::~BatchedEngine() = default;

BatchedEngine::Advance BatchedEngine::advance(uint32_t max_requests) {
    Impl& impl = *impl_;
    Advance out;
    if (max_requests == 0) throw std::runtime_error("max_requests 必须大于 0");
    {
        // 上一轮回填的答案到这里为止：还没被用掉的需求会被补发，因此丢弃是安全的。
        std::lock_guard<std::mutex> lock(impl.cache_mutex);
        impl.fresh_answers.clear();
    }

    // 即使目标已经达成也要继续推进 lane：此时 lane 只会“在局与局之间”停下，
    // 正在进行的局需要被推进到结束，否则它所属的局号永远不会出现在写入序列里，
    // 按局号顺序落盘的写入侧就会永远卡在那一个缺口上（实测踩到过）。
    impl.pump_all(max_requests);

    // 取出这一批请求（受上限约束；多余的留给下一次 advance，Python 会先消化这一批）。
    // 取出的请求登记到 in_flight，submit 用它来定位请求并校验 id。
    std::vector<Impl::Request> batch;
    {
        std::lock_guard<std::mutex> lock(impl.bookkeeping_mutex);
        const size_t take = std::min<size_t>(impl.requests.size(), max_requests);
        batch.assign(impl.requests.begin(), impl.requests.begin() + static_cast<long>(take));
        impl.requests.erase(impl.requests.begin(), impl.requests.begin() + static_cast<long>(take));
        for (const Impl::Request& request : batch) impl.in_flight.emplace(request.id, request);
    }
    out.request_ids.reserve(batch.size());
    out.request_kinds.reserve(batch.size());
    out.boards.resize(batch.size() * kBoardCells);
    for (size_t i = 0; i < batch.size(); i++) {
        out.request_ids.push_back(batch[i].id);
        out.request_kinds.push_back(batch[i].kind);
        std::memcpy(out.boards.data() + i * kBoardCells, batch[i].board, kBoardCells);
    }

    impl.flush_completed();
    {
        std::lock_guard<std::mutex> lock(impl.bookkeeping_mutex);
        out.episodes = std::move(impl.finished_outcomes);
        impl.finished_outcomes.clear();
    }
    out.samples_written = impl.written_samples();
    out.episodes_written = impl.writer ? impl.writer->episodes_written() : 0;
    // done 的条件必须包含“所有 lane 都已收工”：目标达成后 lane 只会在局与局之间停下，
    // 因此还可能有正在跑的局没写完；此时返回 done 会让调用方提前退出，批次就会短于目标条数。
    // 按局数停止的评测运行不看 stop_all（它由样本目标触发），而以“目标局数都已结束”为条件。
    const bool target_reached =
        impl.config.episodes_target > 0
            ? impl.stat_episodes.load() >= impl.config.episodes_target
            : impl.stop_all.load();
    out.done = target_reached && impl.requests.empty() && impl.all_lanes_finished();
    return out;
}

void BatchedEngine::submit(const uint64_t* ids, const float* values, const float* policy_logits,
                           size_t count) {
    Impl& impl = *impl_;
    if (count == 0) return;

    // 先在 bookkeeping 锁下认领请求，再单独在 cache 锁下写入结果：两把锁不嵌套。
    struct PendingPut {
        Board board;
        uint8_t kind;
        uint32_t lane;
        float value;
        float logits[kNumActions];
    };
    std::vector<PendingPut> puts;
    std::vector<uint32_t> touched_lanes;
    {
        std::lock_guard<std::mutex> bookkeeping(impl.bookkeeping_mutex);
        puts.reserve(count);
        for (size_t i = 0; i < count; i++) {
            const uint64_t id = ids[i];
            const auto found = impl.in_flight.find(id);
            if (found == impl.in_flight.end()) {
                // 重复回填或回填了从未发出的 id：属于调用方错误，
                // 必须直接失败——静默忽略会让某条 lane 永远等下去。
                throw std::runtime_error("收到了未知或重复回填的请求 id: " + std::to_string(id));
            }
            const Impl::Request request = found->second;
            impl.in_flight.erase(found);
            PendingPut put;
            std::memcpy(put.board.tiles, request.board, kBoardCells);
            put.kind = request.kind;
            put.lane = request.lane;
            put.value = values[i];
            for (int action = 0; action < kNumActions; action++) {
                put.logits[action] = policy_logits[i * kNumActions + action];
            }
            puts.push_back(put);
            if (request.lane < impl.lanes.size()) {
                Impl::Lane& owner = impl.lanes[request.lane];
                if (owner.outstanding > 0) owner.outstanding--;
            }
            if (std::find(touched_lanes.begin(), touched_lanes.end(), request.lane) == touched_lanes.end()) {
                touched_lanes.push_back(request.lane);
            }
        }
    }
    // 每写入一条结果就立刻尝试结算等待它的挂起模拟，而不是等整批写完再结算。
    //
    // 这不是优化，是正确性要求：推理缓存随时可能把刚写入的条目挤掉（几个高频局面
    // 落在同一个 set 时必然发生），若等整批结束才解析，等待中的模拟就永远看不到自己
    // 的结果，表现为“同一批请求被无限补发”的活锁——这是实测踩到过的形态。
    for (const PendingPut& put : puts) {
        {
            std::lock_guard<std::mutex> cache_lock(impl.cache_mutex);
            if (put.kind == kRequestValue) {
                impl.cache_put(put.board, kRequestValue, put.value, nullptr);
            } else {
                impl.cache_put(put.board, kRequestPolicy, 0.0f, put.logits);
            }
        }
        if (put.lane < impl.lanes.size()) impl.resolve_ready(impl.lanes[put.lane]);
    }
    impl.stat_submitted.fetch_add(count);

    // 再兜底结算一遍：有些模拟需要本批里多条结果才能就绪。
    for (const uint32_t lane_index : touched_lanes) {
        impl.resolve_ready(impl.lanes[lane_index]);
    }
}

EngineStats BatchedEngine::stats() const {
    Impl& impl = *impl_;
    EngineStats stats;
    stats.requests_issued = impl.stat_requests.load();
    stats.requests_submitted = impl.stat_submitted.load();
    stats.leaf_evaluations = impl.stat_leaf.load();
    stats.leaf_from_cache = impl.stat_leaf_cached.load();
    stats.simulations = impl.stat_simulations.load();
    stats.episodes = impl.stat_episodes.load();
    stats.samples = impl.written_samples();
    return stats;
}

BatchedEngine::DebugState BatchedEngine::debug_state() const {
    Impl& impl = *impl_;
    DebugState state;
    std::lock_guard<std::mutex> lock(impl.bookkeeping_mutex);
    state.lanes.reserve(impl.lanes.size());
    for (const Impl::Lane& lane : impl.lanes) {
        LaneStatus status;
        status.episode = lane.episode;
        status.step = lane.step;
        status.simulations_done = lane.simulations_done;
        status.pending = static_cast<uint32_t>(lane.pending.size());
        status.outstanding = lane.outstanding;
        status.finished = lane.finished;
        status.needs_new_episode = lane.needs_new_episode;
        state.lanes.push_back(status);
    }
    {
        std::lock_guard<std::mutex> cache_lock(impl.cache_mutex);
        for (size_t i = 0; i < impl.lanes.size(); i++) impl.fill_missing(impl.lanes[i], state.lanes[i]);
    }
    state.queued_requests = impl.requests.size();
    state.in_flight_requests = impl.in_flight.size();
    state.next_episode = impl.next_episode.load();
    state.next_episode_to_write = impl.next_episode_to_write;
    state.completed_buffered = impl.completed.size();
    for (const auto& entry : impl.completed) state.completed_keys.push_back(entry.first);
    state.samples_produced = impl.samples_produced;
    state.samples_written = impl.written_samples();
    state.stop_all = impl.stop_all.load();
    state.reissues = impl.stat_reissues.load();
    return state;
}

std::string BatchedEngine::finish() {
    Impl& impl = *impl_;
    if (impl.finished) throw std::runtime_error("引擎已经收尾过了");
    if (impl.writer == nullptr) {
        throw std::runtime_error("本次运行关闭了样本收集（评测模式），没有数据批需要收尾");
    }
    impl.stop_all.store(true);
    impl.flush_completed();

    BatchMeta meta;
    meta.batch_id = impl.batch_id;
    meta.policy = "nn";
    meta.simulations = impl.config.decision_mode == kDecisionSearch ? impl.config.simulations : 0;
    meta.temperature_moves = impl.config.temperature_moves;
    meta.seed = impl.config.seed;
    meta.workers = impl.config.lanes;  // 产出这批数据的并发对局数（协议参数，见 ADR 0003）
    meta.samples_target = impl.samples_target;
    meta.model_id = impl.model_id;
    // 这些参数会改变搜索树的演化，因此属于数据生成协议的一部分，必须记录下来。
    meta.params.emplace_back("c_puct", std::to_string(impl.config.c_puct));
    meta.params.emplace_back("lanes", std::to_string(impl.config.lanes));
    meta.params.emplace_back("in_flight_per_lane", std::to_string(impl.config.in_flight_per_lane));
    meta.params.emplace_back("temperature", std::to_string(impl.config.temperature));
    meta.params.emplace_back("cache_entries", std::to_string(impl.config.cache_entries));

    const BatchManifest manifest = impl.writer->finish(meta);
    write_manifest(impl.data_dir, manifest);
    impl.finished = true;
    impl.stats = stats();
    return impl.data_dir + "/manifest.json";
}

}  // namespace r2048
