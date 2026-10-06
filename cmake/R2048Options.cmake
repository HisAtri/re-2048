# 全部构建选项集中在这里，默认值以“本机本地开发”为准。
# 本项目不发布二进制，构建产物只服务这台机器，因此默认值可以偏向开发效率。

option(R2048_BUILD_PYEXT "构建 pybind11 扩展模块 r2048_core" ON)
option(R2048_NATIVE_ARCH "使用 -march=native（产物不可移植到其它 CPU）" OFF)
option(R2048_WERROR "把编译警告视为错误" ON)
option(R2048_ENABLE_SANITIZERS "启用 AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(R2048_DISABLE_CCACHE "禁用 ccache 加速" OFF)
