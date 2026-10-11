cmake_minimum_required(VERSION 3.17)
project(forge_aot_consumer LANGUAGES CXX)
find_package(TaichiForge 0.6.4 CONFIG REQUIRED)
add_executable(aot_runner main.cpp)
target_compile_features(aot_runner PRIVATE cxx_std_17)
target_link_libraries(aot_runner PRIVATE TaichiForge::c_api)
if(WIN32)
    add_custom_command(TARGET aot_runner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:TaichiForge::c_api> $<TARGET_FILE_DIR:aot_runner>)
endif()
