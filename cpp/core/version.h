#pragma once

namespace r2048 {

// 跨语言契约版本。规则语义、动作顺序、棋盘含义、批推理协议中任何一项发生变化，
// 都必须同时抬高这里的版本号：跑谱数据带着产出时的版本落盘，读回时校验，
// 否则会用错误的语义解释旧数据（旧版最贵的坑就是把语义变更与旧数据混在一起）。
constexpr int kRulesVersion = 1;
constexpr int kProtocolVersion = 1;

// 动作顺序是 C++ 与 Python 的硬契约：策略头的 4 个输出、数据记录里的 visits 顺序、
// 缓存里的键都按这个顺序。冻结为与旧版一致，避免历史数据在方向上错位。
enum Action : int {
    kUp = 0,
    kRight = 1,
    kDown = 2,
    kLeft = 3,
};
constexpr int kNumActions = 4;

// 分数归一化尺度：网络输出以“分数 / 1024”为量纲，C++ 搜索内部一律用原始分数，
// 只在跨语言边界上按这个尺度换算，避免两侧各有一套缩放约定。
constexpr float kScoreScale = 1024.0f;

// 棋盘边长与格数。
constexpr int kBoardSide = 4;
constexpr int kBoardCells = kBoardSide * kBoardSide;

}  // namespace r2048
