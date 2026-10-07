#pragma once

#include <string>

namespace r2048 {

// 外部控制通道收到的指令。
//   kNone   没有指令（或指令已被消费过）
//   kPause  在下一个合法停车点停下，并把这一批标记为“已暂停”
//   kStop   同上，但标记为“已停止”
enum class ControlCommand { kNone, kPause, kStop };

std::string control_command_name(ControlCommand command);

// 跑谱是“跑到被叫停为止”的任务，所以必须有一条进程外的控制通道。
//
// 两个通道合一：
//   * **控制文件**（内容是 `pause` 或 `stop`）：C++ 与 Python 侧语义完全一致，
//     可测试、可被脚本与界面写、不需要知道进程号。
//   * **SIGINT / SIGTERM**：等价于 stop。没有它，Ctrl-C 会直接杀掉进程，
//     尾部已产出的数据全丢——而它本可以是一次正常的收尾。
//
// 一致性来自第二条：**一次叫停是一次性的**。指令生效时控制文件被删除。
// 否则 `--continue` 一启动就会读到上次留下的 stop，立刻又停下。
//
// 轮询有节流：主循环每秒可能转几万圈，读文件不能进热路径。
class Control {
public:
    Control() = default;

    // path 为空表示只用信号通道（评测这类短任务用不上控制文件）。
    void set_path(std::string path);
    // 多写者（--join）时设 true：控制文件**不删**，于是每个写者都能读到同一条指令
    // （一次性消费只能停住抢到它的那一个进程，其余会继续跑——那不是用户期望的"停这一批"）。
    void set_sticky(bool sticky) { sticky_ = sticky; }
    const std::string& path() const { return path_; }

    // 安装 SIGINT/SIGTERM 处理。只允许纯 C++ 命令调用：Python 侧有自己的
    // KeyboardInterrupt 语义，两边都装会互相抢。
    static void install_signal_handlers();
    // 信号通道是否被触发过（安装过处理器之后才有意义）。
    static bool signal_received();
    // 清掉信号标记。只给测试用：一个进程里反复起停跑谱时，上一次的标记会污染下一次。
    static void clear_signal();

    // 取一次指令。同一个指令只会返回一次（控制文件被删除，信号标记被清掉）。
    ControlCommand poll();

private:
    bool sticky_ = false;
    std::string path_;
    std::string warned_content_;  // 已经报过错的非法内容（同一句话不重复刷屏）
    double next_poll_at_ = 0.0;   // 节流：下一次允许读文件的时间
};

}  // namespace r2048
