cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 20)
project(vision)

set(LIB_ROOT lib)
set(SRC_DIR ${LIB_ROOT}/src)
set(INCLUDE_DIR ${LIB_ROOT}/include)

find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)

include_directories(
    ${INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

set(SRC_FILES
    ${SRC_DIR}/methods.cpp
    ${SRC_DIR}/generator.cpp
)

add_library(assessment STATIC ${SRC_FILES})

target_link_libraries(assessment 
    PRIVATE 
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
)

install(DIRECTORY ${INCLUDE_DIR}/ 
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

add_executable(gen src/generate.cpp)
target_include_directories(gen PRIVATE lib/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(gen PRIVATE assessment ${OpenCV_LIBS} nlohmann_json::nlohmann_json)

add_executable(eval src/evaluate.cpp)
target_include_directories(eval PRIVATE lib/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(eval PRIVATE assessment ${OpenCV_LIBS} nlohmann_json::nlohmann_json)

add_executable(ass src/assess.cpp)
target_include_directories(ass PRIVATE lib/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ass PRIVATE assessment ${OpenCV_LIBS} nlohmann_json::nlohmann_json)