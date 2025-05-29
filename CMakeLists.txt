cmake_minimum_required(VERSION 3.10)
project(vision)

# Настройка OpenCV
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})