cmake_minimum_required (VERSION 3.8)

project("TalonV2")

file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h" "src/*.hpp")
add_executable(TalonV2Bot ${FILES_SRC})

set_target_properties(TalonV2Bot PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(TalonV2Bot PROPERTIES CXX_STANDARD 20)

set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")

get_filename_component(GIGALEARN_ROOT "${CMAKE_SOURCE_DIR}/../GigaLearnCPP-Leak" ABSOLUTE)

add_subdirectory("${GIGALEARN_ROOT}/GigaLearnCPP" "GigaLearnCPP-build")
target_link_libraries(TalonV2Bot GigaLearnCPP)

add_subdirectory("${GIGALEARN_ROOT}/RLBotCPP" "RLBotCPP-build")
target_link_libraries(TalonV2Bot RLBotCPP)