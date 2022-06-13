file(GLOB_RECURSE JULIUS_SRCS "*.hpp" "*.cpp")
add_library(juliu_cblas SHARED ${JULIUS_SRCS})
target_include_directories(juliu_cblas PRIVATE .)
target_compile_definitions(juliu_cblas PRIVATE -D BUILDING_JULIUS_DLL)