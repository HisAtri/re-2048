# 测试是独立目标，不编进发行二进制：KataGo 把测试链进 katago 让任何一次改动
# 都要重链整个引擎，这里分开之后增量构建才真正便宜。
add_executable(r2048_tests
    test_main.cpp
    rule_vectors.cpp
    test_args.cpp
    test_dataio.cpp
    test_engine.cpp
    test_hash.cpp
    test_rules.cpp
    test_search.cpp
)
target_link_libraries(r2048_tests PRIVATE r2048_core)

# CTest 只负责按套件分组，用例枚举由测试二进制自己完成（见 test_main.cpp）。
foreach(suite rules hash args dataio search engine)
    add_test(NAME ${suite} COMMAND r2048_tests ${suite})
endforeach()

# 跨语言一致性向量由 Python 侧的 pytest 生成并调用本二进制校验，
# 它需要知道二进制路径，因此把测试目标目录告诉 Python 侧。
set(R2048_TEST_BINARY "$<TARGET_FILE:r2048_tests>" PARENT_SCOPE)
