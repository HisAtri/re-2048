# 编译标志：项目自有代码一律高标准；第三方头文件由 SYSTEM include 隔开，
# 不靠降低警告等级来掩盖问题。
#
# 用全局 CMAKE_CXX_FLAGS 而不是 target 级接口：整个仓库的源码都是我们自己的，
# 只有一个 pybind11 模块算外部依赖，全局设置最简单也最不容易漏（与 KataGo 同思路）。

function(r2048_configure_flags)
    if(MSVC)
        set(_warn /W4 /permissive- /utf-8)
    else()
        set(_warn
            -Wall
            -Wextra
            -Wpedantic
            -Wshadow
            -Wno-sign-compare
            -Wcast-qual
            -Wformat=2
            -Wunused
            -Wnon-virtual-dtor
            -Wno-missing-field-initializers
        )
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
            list(APPEND _warn -Wduplicated-branches -Wduplicated-cond -Wlogical-op)
        endif()
    endif()

    if(R2048_NATIVE_ARCH AND NOT MSVC)
        list(APPEND _warn -march=native -mtune=native)
    endif()

    if(R2048_ENABLE_SANITIZERS AND NOT MSVC)
        list(APPEND _warn -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
    endif()

    if(R2048_WERROR AND NOT MSVC)
        list(APPEND _warn -Werror)
    elseif(R2048_WERROR)
        list(APPEND _warn /WX)
    endif()

    # 必须显式用空格连接：CMAKE_CXX_FLAGS 是单个字符串，直接把列表插值进去
    # 会得到 "-Wall;-Wextra;..." 这样一条非法参数，编译器会整体拒收，
    # 连 find_package(Threads) 的探测都会被带崩。
    string(JOIN " " _warn_str ${_warn})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_warn_str}" PARENT_SCOPE)

    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -fno-omit-frame-pointer" PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG" PARENT_SCOPE)
endfunction()
