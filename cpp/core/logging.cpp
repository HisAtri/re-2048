#include "core/logging.h"

#include <cstdarg>
#include <cstdio>
#include <ctime>

#include "core/json.h"

namespace r2048 {

namespace {
constexpr double kRenderIntervalSeconds = 0.2;
}

void log_status(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

void log_error(const char* fmt, ...) {
    std::fputs("错误: ", stderr);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

double monotonic_seconds() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

Progress::Progress(std::string label)
    : label_(std::move(label)),
      done_(0),
      total_(0),
      started_(monotonic_seconds()),
      next_render_at_(0.0),
      rendered_(false) {}

void Progress::set_total(int64_t total) {
    total_ = total;
    next_render_at_ = 0.0;
}

void Progress::set_json(std::string stage, std::string unit) {
    stage_ = std::move(stage);
    unit_ = std::move(unit);
    json_ = true;
}

void Progress::add(int64_t delta) {
    done_ += delta;
    update(done_);
}

void Progress::update(int64_t done) {
    done_ = done;
    const double now = monotonic_seconds();
    if (now < next_render_at_) return;
    next_render_at_ = now + kRenderIntervalSeconds;
    render(false);
}

std::string Progress::format_line() const {
    char line[256];
    // total_ == 0 表示“不限量”（跑到被叫停为止），此时只报累计量而不报百分比：
    // 编一个假的分母会让人以为进度条有意义。
    if (total_ > 0) {
        const double ratio = static_cast<double>(done_) / static_cast<double>(total_);
        std::snprintf(line, sizeof(line), "%s %lld/%lld (%.1f%%)", label_.c_str(),
                      static_cast<long long>(done_), static_cast<long long>(total_), ratio * 100.0);
    } else {
        std::snprintf(line, sizeof(line), "%s %lld", label_.c_str(), static_cast<long long>(done_));
    }
    return line;
}

void Progress::render_json() const {
    // Json 没有浮点：速率取整（样本/秒），ETA 取整秒，-1 表示估不出来。
    const double elapsed = monotonic_seconds() - started_;
    const int64_t rate = elapsed > 0 ? static_cast<int64_t>(static_cast<double>(done_) / elapsed) : 0;
    int64_t eta = -1;
    if (total_ > 0 && rate > 0) eta = (total_ - done_ + rate - 1) / rate;
    Json payload = Json::object();
    payload.set("stage", Json::str(stage_));
    payload.set("done", Json::integer(done_));
    payload.set("total", Json::integer(total_));
    payload.set("unit", Json::str(unit_));
    payload.set("elapsed_ms", Json::integer(static_cast<int64_t>(elapsed * 1000.0)));
    payload.set("rate", Json::integer(rate));
    payload.set("eta_seconds", Json::integer(eta));
    payload.set("line", Json::str(format_line()));
    Json root = Json::object();
    root.set("progress", std::move(payload));
    // stderr：stdout 只留给可被管道消费的最终结果（ADR 0008）。
    std::fprintf(stderr, "%s\n", root.dump(0).c_str());
    std::fflush(stderr);
}

void Progress::render(bool final_render) {
    rendered_ = true;
    if (json_) {
        render_json();
        return;
    }
    const std::string line = format_line();
    if (final_render) {
        std::fprintf(stderr, "\r%s\n", line.c_str());
    } else {
        std::fprintf(stderr, "\r%-100s", line.c_str());
    }
    std::fflush(stderr);
}

void Progress::finish() {
    if (rendered_) render(true);
}

}  // namespace r2048
