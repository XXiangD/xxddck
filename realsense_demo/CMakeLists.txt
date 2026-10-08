cmake_minimum_required(VERSION 3.16)
project(d435i_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 生成编译配置，供 VS Code 使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(realsense2 REQUIRED)
find_package(Threads REQUIRED)

find_package(OpenCV REQUIRED COMPONENTS
    core
    highgui
    imgproc
    dnn
)

set(ONNXRUNTIME_ROOT "/home/humble/onnxruntime-linux-x64-gpu_cuda13-1.30.0"
    CACHE PATH "ONNX Runtime directory"
)

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "找不到 ONNX Runtime 头文件，请检查 ONNXRUNTIME_ROOT"
    )
endif()

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    message(FATAL_ERROR
        "找不到 ONNX Runtime 动态库，请检查 ONNXRUNTIME_ROOT"
    )
endif()

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime_providers_cuda.so")
    message(FATAL_ERROR "当前程序需要 GPU 版 ONNX Runtime，缺少 libonnxruntime_providers_cuda.so")
endif()

set(CUDA_RUNTIME_LIB_DIR "/home/humble/volleyball1/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
    CACHE PATH "CUDA runtime library directory")
set(CUDNN_LIB_DIR "/home/humble/volleyball1/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
    CACHE PATH "cuDNN library directory")

add_library(onnxruntime SHARED IMPORTED)

set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
)

add_executable(diaoyongyolo src/diaoyongyolo.cpp)

target_include_directories(diaoyongyolo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(diaoyongyolo PRIVATE
    realsense2::realsense2
    ${OpenCV_LIBS}
    onnxruntime
    Threads::Threads
)

set_target_properties(diaoyongyolo PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${CUDA_RUNTIME_LIB_DIR};${CUDNN_LIB_DIR}"
)
# CUDA provider 是动态加载的，使用可传递的 RPATH 查找其间接依赖。
target_link_options(diaoyongyolo PRIVATE "LINKER:--disable-new-dtags"
)
