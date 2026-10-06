cmake_minimum_required(VERSION 3.24)
project(r2048 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(R2048Options)
include(R2048Warnings)
include(R2048BuildId)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# 核心库会链进 pybind11 模块，所有目标统一开 PIC，省掉一套特例。
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "构建类型" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo)
endif()

# ccache 是纯白捡的加速：KataGo 没接，我们在 Linux 上顺手接上。
if(NOT R2048_DISABLE_CCACHE)
    find_program(R2048_CCACHE ccache)
    if(R2048_CCACHE)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${R2048_CCACHE}")
        message(STATUS "ccache: ${R2048_CCACHE}")
    endif()
endif()

r2048_configure_flags()
r2048_configure_build_id()

# 产物目录固定：Makefile 与 Python 侧按固定路径取用，不靠猜。
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

message(STATUS "构建类型: ${CMAKE_BUILD_TYPE}")
message(STATUS "构建身份: ${R2048_BUILD_ID}")

enable_testing()
add_subdirectory(cpp)
