add_executable(test_sgemv test_sgemv.cpp)
target_link_libraries(test_sgemv PRIVATE juliu_cblas)