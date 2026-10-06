#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace r2048 {

bool path_exists(const std::string& path);
uint64_t file_size(const std::string& path);
// 目录不存在或存在但为空都算“可用作新输出目录”；非空则拒绝，避免半成品被覆盖。
bool dir_is_usable_as_output(const std::string& path);
void ensure_dir(const std::string& path);
// 为即将写入的文件创建父目录：默认输出路径（例如 $R2048_OUTPUT_DIR/eval/x.jsonl）
// 的父目录在首次运行时并不存在，让用户先手工 mkdir 是没有道理的。
void ensure_parent_dir(const std::string& path);

// 原子落盘：先写 <path>.tmp，成功后 rename 覆盖。
// 跑谱数据的最坏故障是“文件看起来存在但只写了一半”，读回时才在随机位置崩掉；
// 用临时文件 + rename 让“存在”等于“完整”，也让中断留下的 .tmp 一眼可辨。
void write_file_atomic(const std::string& path, const void* data, size_t len);
void write_text_atomic(const std::string& path, const std::string& text);
std::string read_text_file(const std::string& path);

// 缓冲写入器：分片动辄几十 MB，逐条 write 会把时间全花在 syscall 上。
// 构造即打开 <path>.tmp，close() 时 flush 并 rename，中途析构则丢弃临时文件。
class BufferedWriter {
public:
    explicit BufferedWriter(const std::string& path, size_t buffer_bytes = 1u << 20);
    ~BufferedWriter();

    BufferedWriter(const BufferedWriter&) = delete;
    BufferedWriter& operator=(const BufferedWriter&) = delete;

    void write(const void* data, size_t len);
    // 提交：flush + rename。调用后对象不再可用。
    void commit();
    uint64_t bytes_written() const { return bytes_written_; }
    const std::string& path() const { return path_; }

private:
    void flush();

    std::string path_;
    std::string temp_path_;
    std::vector<uint8_t> buffer_;
    size_t used_;
    uint64_t bytes_written_;
    void* file_;
    bool committed_;
};

}  // namespace r2048
