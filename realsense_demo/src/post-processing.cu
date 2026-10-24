#include <cuda_runtime.h>
#include "post-processing.h"
#include <cmath>

// allocatePostprocess(float **output)为输出分配显存
cudaError_t allocatePostprocess(float **output)
{
    if (output == nullptr)
    {
        return cudaErrorInvalidValue;
    }

    // 模型输出：[1, 5, 6]，共 30 个 float。
    const size_t outputBytes = 1 * 5 * 6 * sizeof(float);

    // 第一步：分配显存。
    cudaError_t error = cudaMalloc(
        reinterpret_cast<void **>(output),
        outputBytes);

    if (error != cudaSuccess)
    {
        return error;
    }

    // 第二步：将分配的显存清零。
    error = cudaMemset(*output, 0, outputBytes);

    if (error != cudaSuccess)
    {
        cudaFree(*output);
        *output = nullptr;
    }

    return error;
}

// 保存模型输出结果
cudaError_t saveModelOutput(
    float *output,            // output：刚刚分配的 GPU 显存。
    const float *modelOutput) // modelOutput：模型输出的 GPU 指针。
{
    if (output == nullptr || modelOutput == nullptr)
    {
        return cudaErrorInvalidValue;
    }

    const size_t outputBytes = 1 * 5 * 6 * sizeof(float);

    // 把模型在GPU输出的结果直接保存在GPU
    cudaError_t error = cudaMemcpy(
        output,
        modelOutput,
        outputBytes,
        cudaMemcpyDeviceToDevice);

    return error;
}

// 在主程序循环外调用一次cudaError_t allocateDetections(Detection **detections)

// 为检测框分配显存
cudaError_t allocateDetections(
    Detection **detections // detections指向 GPU 显存的指针，用于保存解析后的检测框
)
{
    if (detections == nullptr)
    {
        return cudaErrorInvalidValue;
    }

    return cudaMalloc(
        reinterpret_cast<void **>(detections),
        5 * sizeof(Detection)); // 分配 5 个检测框的存储空间。
}

// GPU解析检测结果每个线程解析一行检测结果，并保存在Detection
__global__ void parseDetectionsKernel(
    const float *output,
    Detection *detections)
{
    const int index =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= 5)
    {
        return;
    }

    const float *row = output + index * 6;

    //复制边界框和置信度
    detections[index].x1 = row[0];
    detections[index].y1 = row[1];
    detections[index].x2 = row[2];
    detections[index].y2 = row[3];
    detections[index].score = row[4];

    // 模型只有一个类别。
    detections[index].classId = 0;
    // 后续由过滤函数判断
    detections[index].valid = 0;

    detections[index].centerX = 0.0f;
    detections[index].centerY = 0.0f;
}

// 启动核函数，解析检测结果
cudaError_t parseDetections(
    const float *output,
    Detection *detections)
{
    if (output == nullptr || detections == nullptr)
    {
        return cudaErrorInvalidValue;
    }

    parseDetectionsKernel<<<1, 32>>>(
        output,
        detections);

    return cudaGetLastError();
}

// 过滤检测框
__global__ void filterDetectionsKernel(
    Detection *detections,
    // confidenceThreshold 置信度阈值
    float confidenceThreshold)
{
    const int index =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= 5)
    {
        return;
    }

    Detection &box = detections[index];

    // 每帧重新设置，避免沿用上一帧的状态。
    box.valid = 0;

    // 排除无穷大和 NaN。
    if (!isfinite(box.x1) ||
        !isfinite(box.y1) ||
        !isfinite(box.x2) ||
        !isfinite(box.y2) ||
        !isfinite(box.score))
    {
        return;
    }

    // 排除宽、高无效的框。
    if (box.x2 <= box.x1 || box.y2 <= box.y1)
    {
        return;
    }

    // 排除置信度不足的框。
    if (box.score < confidenceThreshold)
    {
        return;
    }

    box.valid = 1;
}

// 启动核函数，过滤检测框
cudaError_t filterDetections(
    Detection *detections,
    float confidenceThreshold)
{
    if (detections == nullptr ||
        !std::isfinite(confidenceThreshold) ||
        confidenceThreshold < 0.0f ||
        confidenceThreshold > 1.0f)
    {
        return cudaErrorInvalidValue;
    }

    filterDetectionsKernel<<<1, 32>>>(
        detections,
        confidenceThreshold);

    return cudaGetLastError();
}

// 在主程序循环外调用一次cudaError_t allocateSelectedDetection(Detection **selected)

// 为最终选中的一个目标分配显存。
cudaError_t allocateSelectedDetection(Detection **selected)
{
    if (selected == nullptr)
    {
        return cudaErrorInvalidValue;
    }

    return cudaMalloc(
        reinterpret_cast<void **>(selected),
        sizeof(Detection));
}

// 选择置信度最高的有效框
__global__ void selectTargetKernel(
    const Detection *detections,
    Detection *selected)
{
    // 只用一个线程扫描 5 个框，避免多个线程同时写结果。
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    Detection best{};
    best.valid = 0;

    for (int i = 0; i < 5; ++i)
    {
        const Detection &box = detections[i];

        // 跳过未通过过滤的框。
        if (box.valid != 1)
        {
            continue;
        }

        // 同分时保留靠前的框。
        if (best.valid == 0 || box.score > best.score)
        {
            best = box;
        }
    }

    // 每帧覆盖结果；没有有效框时，valid 为 0。
    *selected = best;
}

// 启动核函数，选择置信度最高的有效框
cudaError_t selectTarget(
    const Detection *detections,
    Detection *selected)
{
    if (detections == nullptr || selected == nullptr)
    {
        return cudaErrorInvalidValue;
    }

    selectTargetKernel<<<1, 1>>>(
        detections,
        selected);

    return cudaGetLastError();
}

// 将选中框从模型输入坐标还原到原图坐标
__global__ void restoreTargetCoordinatesKernel(
    Detection *selected,
    float scale,
    int padLeft,
    int padTop,
    int imageWidth,
    int imageHeight)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    Detection &box = *selected;

    if (box.valid != 1)
    {
        return;
    }

    // 去掉补边，再撤销缩放。
    const float x1 = (box.x1 - padLeft) / scale;
    const float y1 = (box.y1 - padTop) / scale;
    const float x2 = (box.x2 - padLeft) / scale;
    const float y2 = (box.y2 - padTop) / scale;

    if (!isfinite(x1) || !isfinite(y1) ||
        !isfinite(x2) || !isfinite(y2))
    {
        box.valid = 0;
        return;
    }

    // 保存未裁剪框的中心，供后续读取深度。
    box.centerX = x1 * 0.5f + x2 * 0.5f;
    box.centerY = y1 * 0.5f + y2 * 0.5f;

    // 将绘图边界限制在原图范围内。
    box.x1 = fminf(fmaxf(x1, 0.0f), static_cast<float>(imageWidth));
    box.y1 = fminf(fmaxf(y1, 0.0f), static_cast<float>(imageHeight));
    box.x2 = fminf(fmaxf(x2, 0.0f), static_cast<float>(imageWidth));
    box.y2 = fminf(fmaxf(y2, 0.0f), static_cast<float>(imageHeight));

    // 裁剪后没有面积，说明框与原图没有有效交集。
    box.valid = box.x2 > box.x1 && box.y2 > box.y1;
}

// 启动核函数，选择置信度最高的有效框
cudaError_t restoreTargetCoordinates(
    Detection *selected,
    float scale,
    int padLeft,
    int padTop,
    int imageWidth,
    int imageHeight)
{
    if (selected == nullptr ||
        !std::isfinite(scale) ||
        scale <= 0.0f ||
        imageWidth <= 0 ||
        imageHeight <= 0)
    {
        return cudaErrorInvalidValue;
    }

    restoreTargetCoordinatesKernel<<<1, 1>>>(
        selected,
        scale,
        padLeft,
        padTop,
        imageWidth,
        imageHeight);

    return cudaGetLastError();
}

// 将 GPU 上最终选中的一个检测结果拷回 CPU。
cudaError_t copySelectedToCPU(
    const Detection *gpuSelected,
    Detection *cpuSelected)
{
    if (gpuSelected == nullptr || cpuSelected == nullptr)
    {
        return cudaErrorInvalidValue;
    }

    return cudaMemcpy(
        cpuSelected,       // 目标：CPU
        gpuSelected,       // 来源：GPU
        sizeof(Detection), // 一个检测结果
        cudaMemcpyDeviceToHost);
}
