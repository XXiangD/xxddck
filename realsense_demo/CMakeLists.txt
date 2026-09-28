cmake_minimum_required(VERSION 3.16)
project(d435i_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(realsense2 REQUIRED)

# 图像缩放需要 imgproc，blobFromImage 和 NMSBoxes 需要 dnn
find_package(OpenCV REQUIRED COMPONENTS
    core
    highgui
    imgproc
    dnn
)

# ONNX Runtime C++ 库的解压目录
set(ONNXRUNTIME_ROOT "/home/humble/onnxruntime"
    CACHE PATH "ONNX Runtime directory"
)

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "找不到 ONNX Runtime 头文件，请检查 ONNXRUNTIME_ROOT")
endif()

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    message(FATAL_ERROR "找不到 ONNX Runtime 动态库，请检查 ONNXRUNTIME_ROOT")
endif()

add_library(onnxruntime_cpu SHARED IMPORTED)

set_target_properties(onnxruntime_cpu PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
)

# 源文件路径相对于此 CMakeLists.txt
add_executable(diaoyongyolo src/diaoyongyolo.cpp)

target_include_directories(diaoyongyolo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(diaoyongyolo PRIVATE
    realsense2::realsense2
    ${OpenCV_LIBS}
    onnxruntime_cpu
)

set_target_properties(diaoyongyolo PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
)