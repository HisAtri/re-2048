#include "core/control.h"

#include <csignal>
#include <cstdio>

#include "core/fileio.h"
#include "core/logging.h"

namespace r2048 {

namespace {

// 节流间隔：控制文件是给人按的（或者给脚本写的），200ms 的响应已经远超必要，
// 而主循环一秒可能转几万圈，不节流就会把 read+parse 塞进热路径。
constexpr double kPollIntervalSeconds = 0.2;

// 信号处理器里只能碰 volatile sig_atomic_t 这种平凡对象：别的做法（分配、加锁、
// 打印）都是未定义行为，而这段代码会在 Ctrl-C 的那一刻运行。
volatile std::sig_atomic_t g_stop_requested = 0;

void on_stop_signal(int) {
    g_stop_requested = 1;
}

// 去掉首尾空白：`echo stop > f` 会带上换行，手工编辑时还可能带空格。
std::string trim(const std::string& text) {
    const size_t begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    const size_t end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

}  // namespace

std::string control_command_name(ControlCommand command) {
    switch (command) {
        case ControlCommand::kPause:
            return "pause";
        case ControlCommand::kStop:
            return "stop";
        case ControlCommand::kNone:
            break;
    }
    return "none";
}

void Control::set_path(std::string path) {
    path_ = std::move(path);
    next_poll_at_ = 0.0;
}

void Control::install_signal_handlers() {
    std::signal(SIGINT, on_stop_signal);
    std::signal(SIGTERM, on_stop_signal);
}

bool Control::signal_received() { return g_stop_requested != 0; }

void Control::clear_signal() { g_stop_requested = 0; }

ControlCommand Control::poll() {
    const double now = monotonic_seconds();
    if (now < next_poll_at_) return ControlCommand::kNone;
    next_poll_at_ = now + kPollIntervalSeconds;

    // 信号优先：它不需要读文件，也不该被文件通道的节流拖慢。
    if (g_stop_requested != 0) {
        g_stop_requested = 0;
        return ControlCommand::kStop;
    }
    if (path_.empty()) return ControlCommand::kNone;
    if (!path_exists(path_)) return ControlCommand::kNone;

    const std::string content = trim(read_text_file(path_));
    ControlCommand command = ControlCommand::kNone;
    if (content == "pause") {
        command = ControlCommand::kPause;
    } else if (content == "stop") {
        command = ControlCommand::kStop;
    } else if (!content.empty()) {
        // 拼错的控制词必须说出来：静默忽略会让用户以为“按下了停止”，
        // 而进程还在跑，数据还在长。但只在内容变化时报一次——每 200ms 重复
        // 同一句话会把终端刷成一堵墙，反而没人看得见。
        if (content != warned_content_) {
            warned_content_ = content;
            log_error("控制文件 %s 的内容无法识别: %s（只接受 pause / stop）", path_.c_str(),
                      content.c_str());
        }
    } else {
        warned_content_.clear();
    }
    if (command == ControlCommand::kNone) return ControlCommand::kNone;

    if (sticky_) {
        // 多写者（--join）：控制文件留着，让每个写者都能读到同一条指令。一次性消费只能停住
        // 抢到它的那一个进程，其余会继续跑——而用户按下的意图是"停这一批"。
        return command;
    }
    // 一次性：消费掉它。删除失败不是致命错误（进程照样会停），但要说出来——
    // 下次 --continue 会立刻又被这条指令叫停，用户得知道为什么。
    if (std::remove(path_.c_str()) != 0) {
        log_error("控制文件 %s 已被消费但删除失败：续跑前请手工删除它", path_.c_str());
    }
    return command;
}

}  // namespace r2048
