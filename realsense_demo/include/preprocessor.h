#pragma once
#include <cuda_runtime.h>
#include <cstddef>


// 分配原图显存。
cudaError_t allocateImage(
    unsigned char** gpuImage,
    int width,
    int height);

// 将 CPU 图像上传到 GPU。
cudaError_t uploadImage(
    unsigned char* gpuImage,
    const unsigned char* cpuImage,
    int width,
    int height,
    size_t cpuStride);

// 分配模型输入显存。
cudaError_t allocateInput(
    float** gpuInput,
    int inputWidth,
    int inputHeight);

// 缩放、补边、BGR→RGB、归一化、HWC→CHW。
// scale、padLeft、padTop 通过引用返回，供后处理使用。
cudaError_t preprocessImage(
    const unsigned char* gpuImage,
    int width,
    int height,
    float* gpuInput,
    int inputWidth,
    int inputHeight,
    float& scale,
    int& padLeft,
    int& padTop);