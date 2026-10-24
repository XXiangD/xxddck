#pragma once

#include <cuda_runtime.h>

struct Detection
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int classId;
    // vaild，0-1变量，是否有效
    int valid;
    float centerX;
    float centerY;
};

cudaError_t allocatePostprocess(float **output);

cudaError_t saveModelOutput(
    float *output,
    const float *modelOutput);

cudaError_t allocateDetections(Detection **detections);

cudaError_t parseDetections(
    const float *output,
    Detection *detections);

cudaError_t filterDetections(
    Detection *detections,
    float confidenceThreshold);

cudaError_t allocateSelectedDetection(Detection **selected);

cudaError_t selectTarget(
    const Detection *detections,
    Detection *selected);

cudaError_t restoreTargetCoordinates(
    Detection* selected,
    float scale,
    int padLeft,
    int padTop,
    int imageWidth,
    int imageHeight);

cudaError_t copySelectedToCPU(
    const Detection* gpuSelected,
    Detection* cpuSelected);