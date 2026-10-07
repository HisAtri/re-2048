#pragma once

#include <cstdint>
#include <string>

namespace r2048 {

// 全部输出到 stderr：stdout 留给可被管道消费的结果（JSONL、清单路径等），
// 这样 `r2048 evaluate ... --jsonl -` 之类重定向不会把进度行混进数据里。
void log_status(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void log_error(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

// 进度渲染：同一行原地刷新，最多每 200ms 或整百分比变化时写一次。
// 它只是给终端看的视图，完整记录由 log_status 打出的行构成——
// 旧版把进度解析出来的控制台当成第二套仪器，结果两边数字对不上。
//
// 两种呈现方式（`set_json`）：
//   * 默认（人看）——`\r` 覆盖式一行，终端里最舒服；
//   * `--progress-json`——每行一条 JSON 写到 stderr，供父进程解析（脚本、界面）。
//     两者产出的 `line` 字段完全相同，所以“人看到的”与“程序读到的”是同一句话，
//     不存在两套数字对不上的可能。stdout 始终只留给最终结果。
class Progress {
public:
    explicit Progress(std::string label);

    void set_total(int64_t total);
    // stage/unit 用于结构化输出（见 jobs.py 的 PROGRESS_KEYS）；
    // 不调用则保持人类可读的单行渲染。
    void set_json(std::string stage, std::string unit);
    void update(int64_t done);
    void add(int64_t delta);
    void finish();

private:
    void render(bool final_render);
    std::string format_line() const;
    void render_json() const;

    std::string label_;
    std::string stage_;
    std::string unit_;
    bool json_ = false;
    int64_t done_;
    int64_t total_;
    double started_;
    double next_render_at_;
    bool rendered_;
};

// 单调时钟秒数，用于吞吐统计。
double monotonic_seconds();

}  // namespace r2048
