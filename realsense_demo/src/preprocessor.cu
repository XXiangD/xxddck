#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "preprocessor.h"
#include <algorithm>
#include <cmath>

cudaError_t allocateImage(unsigned char** gpuImage,int width,int height){

    //cuda第一步
    size_t bytes = static_cast<size_t>(width) * height * 3;

    cudaError_t error = cudaMalloc(reinterpret_cast<void**>(gpuImage),bytes);

    if (error != cudaSuccess){
        return error;
    }

    //cuda第二步
    error = cudaMemset(*gpuImage,0,bytes);

    if (error != cudaSuccess) {
        cudaFree(*gpuImage);                //清空地址
        *gpuImage = nullptr;                //放飞指针
    }


    return error;
}

// 将 CPU 图像上传到已经分配好的 GPU 显存。
// 每帧调用一次。
cudaError_t uploadImage(
    unsigned char* gpuImage,
    const unsigned char* cpuImage,
    int width,
    int height,
    size_t cpuStride)
{
    if (gpuImage == nullptr ||
        cpuImage == nullptr ||
        width <= 0 ||
        height <= 0)
    {
        return cudaErrorInvalidValue;
    }

    const size_t rowBytes = static_cast<size_t>(width) * 3;

    if (cpuStride < rowBytes)
    {
        return cudaErrorInvalidValue;
    }

    // 按行复制，兼容图像每行末尾存在填充字节的情况。
    return cudaMemcpy2D(
        gpuImage,                 // 目标：GPU 显存
        rowBytes,                 // GPU 每行字节数
        cpuImage,                 // 来源：CPU 图像
        cpuStride,                // CPU 每行实际字节数
        rowBytes,                 // 每行复制多少字节
        height,                   // 复制多少行
        cudaMemcpyHostToDevice);  // CPU → GPU
}

// GPU 预处理核函数。
// 完成：缩放、补边、BGR→RGB、归一化、HWC→CHW。
__global__ void preprocessKernel(
    const unsigned char* gpuImage,
    int width,
    int height,
    float* gpuInput,
    int inputWidth,
    int inputHeight,
    int resizedWidth,
    int resizedHeight,
    int padLeft,
    int padTop)
{
    // 当前线程处理的输出像素坐标。
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= inputWidth || y >= inputHeight)
    {
        return;
    }

    // 默认补边颜色。
    float b = 114.0f;
    float g = 114.0f;
    float r = 114.0f;

    // 去除补边后的坐标。
    const int rx = x - padLeft;
    const int ry = y - padTop;

    // 实际图像区域内才读取原图。
    if (rx >= 0 && rx < resizedWidth &&
        ry >= 0 && ry < resizedHeight)
    {
        // 将输出像素中心映射到原图。
        float sx =
            (rx + 0.5f) * static_cast<float>(width) /
            resizedWidth - 0.5f;

        float sy =
            (ry + 0.5f) * static_cast<float>(height) /
            resizedHeight - 0.5f;

        sx = fminf(
            fmaxf(sx, 0.0f),
            static_cast<float>(width - 1));

        sy = fminf(
            fmaxf(sy, 0.0f),
            static_cast<float>(height - 1));

        // 双线性插值使用的四个像素。
        const int x0 = static_cast<int>(floorf(sx));
        const int y0 = static_cast<int>(floorf(sy));

        const int x1 =
            x0 + 1 < width ? x0 + 1 : width - 1;

        const int y1 =
            y0 + 1 < height ? y0 + 1 : height - 1;

        //相对于左上角像素 (x0, y0) 的偏移量
        const float dx = sx - x0;
        const float dy = sy - y0;

        //四个像素的颜色权重
        const float w00 = (1.0f - dx) * (1.0f - dy);
        const float w01 = dx * (1.0f - dy);
        const float w10 = (1.0f - dx) * dy;
        const float w11 = dx * dy;

        // uploadImage() 上传后的原图是连续的 BGR 数据。
        const unsigned char* p00 =
            gpuImage + (y0 * width + x0) * 3;

        const unsigned char* p01 =
            gpuImage + (y0 * width + x1) * 3;

        const unsigned char* p10 =
            gpuImage + (y1 * width + x0) * 3;

        const unsigned char* p11 =
            gpuImage + (y1 * width + x1) * 3;

        b = p00[0] * w00 + p01[0] * w01 +
            p10[0] * w10 + p11[0] * w11;

        g = p00[1] * w00 + p01[1] * w01 +
            p10[1] * w10 + p11[1] * w11;

        r = p00[2] * w00 + p01[2] * w01 +
            p10[2] * w10 + p11[2] * w11;
    }

    const int index = y * inputWidth + x;
    const int planeSize = inputWidth * inputHeight;

    // RGB 顺序，每个通道单独连续存放，并除以 255。
    gpuInput[index] = r / 255.0f;
    gpuInput[planeSize + index] = g / 255.0f;
    gpuInput[2 * planeSize + index] = b / 255.0f;
}


// 分配预处理结果的显存。
// 在循环外调用一次。
cudaError_t allocateInput(
    float** gpuInput,
    int inputWidth,
    int inputHeight)
{
    if (gpuInput == nullptr ||
        inputWidth <= 0 ||
        inputHeight <= 0)
    {
        return cudaErrorInvalidValue;
    }

    *gpuInput = nullptr;

    const size_t bytes =
        static_cast<size_t>(inputWidth) *
        inputHeight * 3 * sizeof(float);

    return cudaMalloc(
        reinterpret_cast<void**>(gpuInput),
        bytes);
}


// 启动 GPU 预处理。
// 每帧先调用 uploadImage()，再调用这个函数。
cudaError_t preprocessImage(
    const unsigned char* gpuImage,
    int width,
    int height,
    float* gpuInput,
    int inputWidth,
    int inputHeight,
    float& scale,
    int& padLeft,
    int& padTop)
{
    if (gpuImage == nullptr ||
        gpuInput == nullptr ||
        width <= 0 ||
        height <= 0 ||
        inputWidth <= 0 ||
        inputHeight <= 0)
    {
        return cudaErrorInvalidValue;
    }

    // 计算缩放比例。
    scale = std::min(
        inputWidth / static_cast<float>(width),
        inputHeight / static_cast<float>(height));

    const int resizedWidth =
        static_cast<int>(std::round(width * scale));

    const int resizedHeight =
        static_cast<int>(std::round(height * scale));

    if (resizedWidth <= 0 || resizedHeight <= 0)
    {
        return cudaErrorInvalidValue;
    }

    padLeft = (inputWidth - resizedWidth) / 2;
    padTop = (inputHeight - resizedHeight) / 2;

    // 每个线程块包含 16×16 个线程。
    const dim3 block(16, 16);

    const dim3 grid(
        (inputWidth + block.x - 1) / block.x,
        (inputHeight + block.y - 1) / block.y);

    // 启动核函数，像素计算在 GPU 上执行。
    preprocessKernel<<<grid, block>>>(
        gpuImage,
        width,
        height,
        gpuInput,
        inputWidth,
        inputHeight,
        resizedWidth,
        resizedHeight,
        padLeft,
        padTop);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        return error;
    }

    // 等待预处理完成，之后模型才能读取 gpuInput。
    return cudaDeviceSynchronize();
}
