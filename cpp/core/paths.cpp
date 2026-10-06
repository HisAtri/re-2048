#include "core/paths.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>

namespace r2048 {

std::string output_root() {
    const char* from_env = std::getenv("R2048_OUTPUT_DIR");
    if (from_env != nullptr && from_env[0] != '\0') return from_env;
    return "output";
}

std::string default_data_dir(const std::string& batch_id) { return output_root() + "/data/" + batch_id; }

std::string default_eval_dir() { return output_root() + "/eval"; }

std::string make_batch_id(const std::string& policy, uint32_t simulations) {
    std::time_t now = std::time(nullptr);
    std::tm local{};
    localtime_r(&now, &local);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &local);

    std::string id = policy;
    if (simulations > 0) id += "-s" + std::to_string(simulations);
    id += "-";
    id += stamp;
    return id;
}

}  // namespace r2048
