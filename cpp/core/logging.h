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
class Progress {
public:
    explicit Progress(std::string label);

    void set_total(int64_t total);
    void update(int64_t done);
    void add(int64_t delta);
    void finish();

private:
    void render(bool final_render);

    std::string label_;
    int64_t done_;
    int64_t total_;
    double next_render_at_;
    bool rendered_;
};

// 单调时钟秒数，用于吞吐统计。
double monotonic_seconds();

}  // namespace r2048
