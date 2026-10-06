# 源码列表显式列举，不用 GLOB：新增文件必须显式出现在这里，改动可被 code review 看见，
# 也不会出现“删掉的文件还在库里被编译”的情况（KataGo 同样坚持显式列表）。

set(R2048_CORE_SOURCES
    core/args.cpp
    core/fileio.cpp
    core/hash.cpp
    core/json.cpp
    core/logging.cpp
    core/paths.cpp
    dataio/sample.cpp
    dataio/shard.cpp
    game/board.cpp
    game/rules.cpp
    game/tables.cpp
    program/episode.cpp
    program/evaluate.cpp
    program/generate.cpp
    program/verify.cpp
    search/batched_engine.cpp
    search/rollout_mcts.cpp
)

add_library(r2048_core STATIC ${R2048_CORE_SOURCES})
target_include_directories(r2048_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"        # 让 #include "core/rng.h" / "game/board.h" 生效
        "${CMAKE_BINARY_DIR}/generated"      # 生成的 r2048_build_id.h
)

find_package(Threads REQUIRED)
target_link_libraries(r2048_core PUBLIC Threads::Threads)

add_executable(r2048
    main.cpp
    command/bench.cpp
    command/evaluate.cpp
    command/generate.cpp
    command/verify.cpp
)
target_link_libraries(r2048 PRIVATE r2048_core)

if(R2048_BUILD_PYEXT)
    # 扩展模块的导入名是顶层模块 r2048_core（不是 r2048._core）：
    # 若把 .so 放进 python 包的目录里，它会和可编辑安装的包路径互相遮蔽，
    # 而顶层模块名让两者互不干扰，同时源码树里不需要出现任何二进制产物。
    set(R2048_PYEXT_NAME "r2048_core")
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
    if(NOT pybind11_DIR AND NOT DEFINED PYBIND11_FINDPYTHON)
        message(FATAL_ERROR
            "找不到 pybind11 的 CMake 配置。请通过 `make configure` 构建（它会传 "
            "-Dpybind11_DIR=...），或手工传 -Dpybind11_DIR=$(python -m pybind11 --cmakedir)。\n"
            "如果暂时不需要 Python 扩展，可用 -DR2048_BUILD_PYEXT=OFF 只构建 CLI 与测试。")
    endif()
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(r2048_pyext MODULE ext/bindings.cpp)
    set_target_properties(r2048_pyext PROPERTIES
        OUTPUT_NAME "${R2048_PYEXT_NAME}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pyext"
    )
    target_link_libraries(r2048_pyext PRIVATE r2048_core)
endif()

add_subdirectory(tests)
