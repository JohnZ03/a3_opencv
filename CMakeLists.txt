cmake_minimum_required(VERSION 3.21)
project(a3_opencv_base)

set(CMAKE_CXX_STANDARD 23)

# 设置OpenCV目录所在
set(OpenCV_DIR "/home/haotian/Downloads/build")

# 搜索OpenCV目录
find_package(OpenCV REQUIRED)
# 添加OpenCV头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(a3_opencv_base main.cpp)

# 链接OpenCV库文件
target_link_libraries(a3_opencv_base ${OpenCV_LIBS})
