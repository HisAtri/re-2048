#include <cstdio>
#include <map>
#include <set>
#include <string>

#include "core/fileio.h"
#include "core/hash.h"
#include "core/rng.h"
#include "game/board.h"
#include "test_util.h"

namespace r2048::tests {

namespace {

const char* kShaEmpty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
const char* kShaAbc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
const char* kShaLong =
    "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";  // "abcdbcde..."
const char* kShaMillionA = "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";

}  // namespace

TEST(hash_sha256_known_vectors) {
    expect_contains(sha256_hex("", 0), kShaEmpty, "空串摘要");
    expect_equal_u64(sha256_hex("", 0).size(), 64, "摘要长度");
    expect_contains(sha256_hex("abc", 3), kShaAbc, "abc 摘要");

    const std::string long_input = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    expect_contains(sha256_hex(long_input.data(), long_input.size()), kShaLong, "56 字节输入摘要");

    // 跨块边界与两轮以上的输入：100 万个 'a'。
    std::string million(1000000, 'a');
    expect_contains(sha256_hex(million.data(), million.size()), kShaMillionA, "100 万字节摘要");
}

TEST(hash_sha256_streaming_matches_oneshot) {
    // 分块喂入必须与一次性喂入等价：manifest 是按块算哈希的，流式路径出错会
    // 让“数据完整”这个结论变成假的。
    const std::string text = "r2048 streaming hash check: 分块与一次性结果必须一致。";
    const std::string oneshot = sha256_hex(text.data(), text.size());

    for (size_t chunk : {size_t(1), size_t(3), size_t(63), size_t(64), size_t(65), size_t(7)}) {
        Sha256 hasher;
        size_t offset = 0;
        while (offset < text.size()) {
            const size_t take = std::min(chunk, text.size() - offset);
            hasher.update(text.data() + offset, take);
            offset += take;
        }
        uint8_t digest[32];
        hasher.finish(digest);
        expect_contains(to_hex(digest, 32), oneshot, "分块哈希等于一次性哈希");
    }

    // 文件路径与内存路径必须一致。
    const std::string path = "r2048_hash_test.tmp";
    write_text_atomic(path, text);
    expect_contains(sha256_hex_file(path), oneshot, "文件哈希等于内存哈希");
    std::remove(path.c_str());
}

TEST(hash_board_hash_distinct) {
    // 棋盘哈希只用于缓存索引，这里检查两个必要条件：相同棋盘同哈希、
    // 不同棋盘在样本范围内不碰撞。注意随机生成的棋盘本身会大量重复
    // （空格数越少重复越多），所以碰撞必须按“棋盘不同”判定，而不是按哈希不同。
    std::map<uint64_t, Board> seen;
    Rng rng(777);
    int collisions = 0;
    int duplicates = 0;
    const int samples = 200000;
    for (int i = 0; i < samples; i++) {
        Board board = Board::empty();
        const int fill = 1 + static_cast<int>(rng.below(16));
        for (int placed = 0; placed < fill; placed++) {
            board.tiles[rng.below(16)] = static_cast<uint8_t>(1 + rng.below(12));
        }
        const Board copy = board;
        expect_equal_u64(board.hash(), copy.hash(), "相同棋盘哈希一致");

        const auto found = seen.find(board.hash());
        if (found == seen.end()) {
            seen.emplace(board.hash(), board);
        } else if (found->second == board) {
            duplicates++;
        } else {
            collisions++;
        }
    }
    expect_true(duplicates > 0, "样本里应当存在重复棋盘（否则这个用例没有区分能力）");
    // 20 万条样本落在 2^64 空间里，期望真实碰撞数约 1e-9。
    expect_equal_u64(collisions, 0, "不同棋盘之间不应出现哈希碰撞");
}

}  // namespace r2048::tests
