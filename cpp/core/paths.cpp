#include "core/paths.h"
#include "core/fileio.h"

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <stdexcept>

namespace r2048 {

std::string output_root() {
    const char* from_env = std::getenv("R2048_OUTPUT_DIR");
    if (from_env != nullptr && from_env[0] != '\0') return from_env;
    return "output";
}

std::string default_data_dir(const std::string& batch_id) {
    return output_root() + "/data/" + batch_id;
}

std::string default_eval_dir() {
    return output_root() + "/eval";
}

std::string make_batch_id(const std::string& policy, uint32_t simulations) {
    std::time_t now = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&now, &utc);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d", &utc);

    // 与 Python identity.py 共用锁文件和十进制计数器。只锁分配过程，跑谱并行度不受影响。
    const std::string directory = output_root() + "/.identities/data";
    ensure_dir(directory);
    const int fd = ::open((directory + "/lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0644);
    if (fd < 0) throw std::runtime_error("无法打开数据身份分配锁");
    struct CloseFd {
        int fd;
        ~CloseFd() { ::close(fd); }
    } close_fd{fd};
    if (::flock(fd, LOCK_EX) != 0) throw std::runtime_error("无法锁定数据身份分配器");
    const std::string counter = directory + "/" + stamp + ".counter";
    const int previous = path_exists(counter) ? std::stoi(read_text_file(counter)) : 0;
    if (previous < 0 || previous >= 999) throw std::runtime_error("数据批当日三位流水号无效或已耗尽");
    const int number = previous + 1;
    write_text_atomic(counter, std::to_string(number) + "\n");
    char sequence[8];
    std::snprintf(sequence, sizeof(sequence), "%03d", number);

    std::string id = policy;
    id += "-s" + std::to_string(simulations);
    id += "-";
    id += stamp;
    id += sequence;
    return id;
}

}  // namespace r2048
