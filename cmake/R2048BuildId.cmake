# 构建身份：跑谱数据必须能自证“是哪一份二进制产出的”，因此把 git 描述、
# 编译器与生效标志的哈希一起编进二进制。
#
# 用 configure_file 而不是每次构建都跑一遍脚本：configure_file 在内容不变时
# 不更新文件时间戳，因此不会因为“重新 configure 了一次”而让所有目标重编。
# 代价是 build id 反映的是最近一次 configure 时的仓库状态（git 提交后需要重新
# configure 才会更新），这一点写在 docs/BUILD.md 里。

function(r2048_configure_build_id)
    set(_desc "unknown")
    find_package(Git QUIET)
    if(GIT_FOUND)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_SOURCE_DIR}" describe --always --dirty --tags
            OUTPUT_VARIABLE _git_desc
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _git_rc
        )
        if(_git_rc EQUAL 0 AND NOT _git_desc STREQUAL "")
            set(_desc "${_git_desc}")
        endif()
    endif()

    string(SHA1 _flags_hash "${CMAKE_CXX_FLAGS}|${CMAKE_BUILD_TYPE}|${CMAKE_CXX_COMPILER_VERSION}")
    string(SUBSTRING "${_flags_hash}" 0 12 _flags_hash)

    set(R2048_BUILD_ID "${_desc}+${CMAKE_CXX_COMPILER_ID}${CMAKE_CXX_COMPILER_VERSION}.${_flags_hash}")
    set(R2048_BUILD_GIT_DESC "${_desc}")
    set(R2048_BUILD_TYPE "${CMAKE_BUILD_TYPE}")

    configure_file(
        "${CMAKE_SOURCE_DIR}/cpp/core/r2048_build_id.h.in"
        "${CMAKE_BINARY_DIR}/generated/r2048_build_id.h"
        @ONLY
    )

    # 调用方也要能读到构建身份（用于打印与产物记录），因此显式回填父作用域。
    set(R2048_BUILD_ID "${R2048_BUILD_ID}" PARENT_SCOPE)
    set(R2048_BUILD_GIT_DESC "${R2048_BUILD_GIT_DESC}" PARENT_SCOPE)
endfunction()
