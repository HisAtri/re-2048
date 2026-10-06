#include "core/logging.h"

#include <cstdarg>
#include <cstdio>
#include <ctime>

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
    : label_(std::move(label)), done_(0), total_(0), next_render_at_(0.0), rendered_(false) {}

void Progress::set_total(int64_t total) {
    total_ = total;
    next_render_at_ = 0.0;
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

void Progress::render(bool final_render) {
    char line[256];
    if (total_ > 0) {
        const double ratio = static_cast<double>(done_) / static_cast<double>(total_);
        std::snprintf(line, sizeof(line), "%s %lld/%lld (%.1f%%)", label_.c_str(),
                      static_cast<long long>(done_), static_cast<long long>(total_), ratio * 100.0);
    } else {
        std::snprintf(line, sizeof(line), "%s %lld", label_.c_str(), static_cast<long long>(done_));
    }
    if (final_render) {
        std::fprintf(stderr, "\r%s\n", line);
    } else {
        std::fprintf(stderr, "\r%-100s", line);
    }
    std::fflush(stderr);
    rendered_ = true;
}

void Progress::finish() {
    if (rendered_) render(true);
}

}  // namespace r2048
