#pragma once
#include <cuda_runtime.h>

cudaError_t allocateImage(
    unsigned char** gpuImage,
    int width,
    int height);