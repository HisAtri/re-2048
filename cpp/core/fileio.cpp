#include "core/fileio.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace r2048 {

bool path_exists(const std::string& path) { return std::filesystem::exists(path); }

uint64_t file_size(const std::string& path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) throw std::runtime_error("无法获取文件大小 " + path + ": " + ec.message());
    return size;
}

bool dir_is_usable_as_output(const std::string& path) {
    if (!std::filesystem::exists(path)) return true;
    if (!std::filesystem::is_directory(path)) return false;
    return std::filesystem::directory_iterator(path) == std::filesystem::directory_iterator();
}

void ensure_dir(const std::string& path) {
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    if (ec && !std::filesystem::is_directory(path)) {
        throw std::runtime_error("无法创建目录 " + path + ": " + ec.message());
    }
}

void ensure_parent_dir(const std::string& path) {
    const size_t slash = path.find_last_of('/');
    if (slash == std::string::npos || slash == 0) return;
    ensure_dir(path.substr(0, slash));
}

void write_file_atomic(const std::string& path, const void* data, size_t len) {
    const std::string temp_path = path + ".tmp";
    FILE* file = std::fopen(temp_path.c_str(), "wb");
    if (file == nullptr) throw std::runtime_error("无法写入文件: " + temp_path);
    if (len > 0 && std::fwrite(data, 1, len, file) != len) {
        std::fclose(file);
        std::remove(temp_path.c_str());
        throw std::runtime_error("写入文件失败: " + temp_path);
    }
    if (std::fclose(file) != 0) {
        std::remove(temp_path.c_str());
        throw std::runtime_error("关闭文件失败: " + temp_path);
    }
    std::error_code ec;
    std::filesystem::rename(temp_path, path, ec);
    if (ec) {
        std::remove(temp_path.c_str());
        throw std::runtime_error("替换文件失败 " + path + ": " + ec.message());
    }
}

void write_text_atomic(const std::string& path, const std::string& text) {
    write_file_atomic(path, text.data(), text.size());
}

std::string read_text_file(const std::string& path) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) throw std::runtime_error("无法读取文件: " + path);
    std::string out;
    char buffer[1 << 16];
    size_t read = 0;
    while ((read = std::fread(buffer, 1, sizeof(buffer), file)) > 0) out.append(buffer, read);
    const bool failed = std::ferror(file) != 0;
    std::fclose(file);
    if (failed) throw std::runtime_error("读取文件失败: " + path);
    return out;
}

BufferedWriter::BufferedWriter(const std::string& path, size_t buffer_bytes)
    : path_(path),
      temp_path_(path + ".tmp"),
      buffer_(buffer_bytes),
      used_(0),
      bytes_written_(0),
      file_(nullptr),
      committed_(false) {
    FILE* file = std::fopen(temp_path_.c_str(), "wb");
    if (file == nullptr) throw std::runtime_error("无法写入文件: " + temp_path_);
    file_ = file;
}

BufferedWriter::~BufferedWriter() {
    if (file_ != nullptr) {
        std::fclose(static_cast<FILE*>(file_));
        if (!committed_) std::remove(temp_path_.c_str());
    }
}

void BufferedWriter::flush() {
    if (used_ == 0) return;
    FILE* file = static_cast<FILE*>(file_);
    if (std::fwrite(buffer_.data(), 1, used_, file) != used_) {
        throw std::runtime_error("写入文件失败: " + temp_path_);
    }
    used_ = 0;
}

void BufferedWriter::write(const void* data, size_t len) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    bytes_written_ += len;
    while (len > 0) {
        const size_t space = buffer_.size() - used_;
        if (space == 0) {
            flush();
            continue;
        }
        const size_t take = len < space ? len : space;
        std::memcpy(buffer_.data() + used_, bytes, take);
        used_ += take;
        bytes += take;
        len -= take;
    }
}

void BufferedWriter::commit() {
    flush();
    FILE* file = static_cast<FILE*>(file_);
    if (std::fflush(file) != 0 || std::fclose(file) != 0) {
        file_ = nullptr;
        std::remove(temp_path_.c_str());
        throw std::runtime_error("关闭文件失败: " + temp_path_);
    }
    file_ = nullptr;
    std::error_code ec;
    std::filesystem::rename(temp_path_, path_, ec);
    if (ec) {
        std::remove(temp_path_.c_str());
        throw std::runtime_error("替换文件失败 " + path_ + ": " + ec.message());
    }
    committed_ = true;
}

}  // namespace r2048
