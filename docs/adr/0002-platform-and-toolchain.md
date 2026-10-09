# ADR 0002：平台与工具链

- 状态：已采纳
- 日期：2026-10-06

## 背景

开发环境是 WSL2（Ubuntu 22.04，内核 6.6.87.2）内的 Linux；仓库位于 `/mnt/e`，
即 Windows E: 盘的 9p/drvfs 挂载。旧版是 Windows + MSVC + PowerShell 的形态，
其构建文档里能读到大量与平台搏斗的痕迹（VS 环境变量导出、MinGW DLL 冲突、
把 `TEMP` 重定向进仓库）。本项目明确以 Linux 为第一平台，不应重复这类代价。

## 决策

1. **Linux（WSL2）是第一且唯一承诺支持的构建平台**。CMake 保持可移植、代码不写平台特例，
   但不提供、也不维护 Windows 构建脚本；不引入 MSVC 专用分支。
2. **工具链**：GCC 13（`-std=c++20`）、CMake ≥ 3.24、Ninja、ccache。cmake/ninja 用
   `uv tool install` 安装（无需 sudo，版本比 22.04 自带的 3.22 新），ccache 走 apt。
   `make env-check` 负责检测并打印缺失项的安装命令。
3. **构建目录和虚拟环境默认放在本地 ext4**（`$HOME/r2048/{build,venv}`）；数据与模型
   默认放在仓库的 `output/`（见 ADR 0046）。`BUILD`、`VENV` 和 `R2048_OUTPUT_DIR` 可覆盖。
4. **CUDA 不进入 CMake**：GPU 只通过 PyTorch 使用。CUDA toolkit（本机 13.0）与
   `/usr/lib/wsl/lib` 的直通驱动仅用于诊断。未来若真要做 C++ 内置推理后端，
   再按 KataGo 的 `USE_BACKEND` 形态单独引入，并保留“无后端也能构建”的默认。
5. **git 提交后需要重新 configure 才会更新构建身份头**：构建身份用 `configure_file` 生成
   （内容不变时不动时间戳，因此不会造成无谓重编）。`make build` 每次都先 configure，
   所以日常使用中这条不会成为问题。

## 理由

* 9p 挂载对 C++ 编译（大量小文件写入）与自对弈落盘（GB 级顺序写）都明显慢于本地 ext4；
  因此构建与虚拟环境外置。数据和模型则以项目内归档易用性优先，路径仍可重定向。
* 仓库仍在 `/mnt/e`，Windows 侧的编辑器与备份照旧可用；高频构建文件留在 ext4，
  实验产物按 ADR 0046 统一放在项目 `output/`。
* 不引入 MSVC 分支：省掉的是两套编译器行为差异、`/arch:AVX2` 之类的第二套标志、
  以及每加一个 C++20 特性都要考虑 MSVC 支持度的负担。
* CMake 里不出现 CUDA，是把 KataGo 最大的复杂度来源（7 个后端、cuDNN/TensorRT 探测、
  约 200 行 ROCm 工具链定位）整体排除在外。事实是：所有张量计算都在 PyTorch 里，
  C++ 侧没有任何一行需要 nvcc。

## 替代方案

* **同时支持 Windows/MSVC**：旧版就是这条路，代价是双工具链、双标志集、
  以及“改了要在哪台机器重新编译”的持续分心。当前收益为零。
* **把整个仓库搬到 ext4、用 WSL 内的编辑器开发**：性能最好，但会改变现有工作方式；
  折中方案（源码留 9p、产物走 ext4）能拿到几乎全部收益。
* **`-march=native` 默认开启**：产物不可移植，且本机就是唯一目标机器，
  因此由 Makefile 的 build 目标显式开启（`-DR2048_NATIVE_ARCH=ON`），
  CMake 默认关闭，避免“拷到别的机器上直接崩”这类沉默故障。

## 验证方式

* `make env-check` 在本机输出全部工具路径与三个目录的实际取值。
* `make build` 在 16 核机器上从零构建约几十秒（含 ccache 命中时更快）。
* 编译产物为 `build/bin/r2048`、`build/lib/libr2048_core.a`、
  `build/pyext/r2048_core.cpython-312-x86_64-linux-gnu.so`。
