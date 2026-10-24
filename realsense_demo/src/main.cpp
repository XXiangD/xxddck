#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <exception>
#include <yuce.h>
#include <preprocessor.h>
#include <post-processing.h>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <cmath>

// CUDA 调用失败时抛出异常，交给现有异常处理代码处理。
static void checkCuda(cudaError_t error, const char *operation)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

// 自动释放显存，正常退出和异常退出都会执行。
struct GpuBuffers
{
    unsigned char *image = nullptr;
    float *input = nullptr;
    float *output = nullptr;
    Detection *detections = nullptr;
    Detection *selected = nullptr;

    int imageWidth = 0;
    int imageHeight = 0;

    GpuBuffers() = default;
    GpuBuffers(const GpuBuffers &) = delete;
    GpuBuffers &operator=(const GpuBuffers &) = delete;

    ~GpuBuffers()
    {
        if (input)
            cudaFree(input);

        if (image)
            cudaFree(image);

        if (output)
            cudaFree(output);

        if (detections)
            cudaFree(detections);

        if (selected)
            cudaFree(selected);
    }

    // 第一帧分配原图显存；尺寸不变时重复使用。
    void ensureImageSize(int width, int height)
    {
        if (image &&
            width == imageWidth &&
            height == imageHeight)
        {
            return;
        }

        unsigned char *newImage = nullptr;
        (
            allocateImage(&newImage, width, height),
            "allocateImage");

        if (image)
            cudaFree(image);

        image = newImage;
        imageWidth = width;
        imageHeight = height;
    }
};

int runApplication(int argc, char **argv)
{
    // 创建 ONNX Runtime 的运行环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "volleyball");

    // 创建配置对象，用来设置图优化、执行设备、线程等参数
    Ort::SessionOptions sessionOptions;

    // 启用全部可用的图优化，例如删除冗余节点、合并部分算子，从而减少计算开销，提高推理效率。
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 使用第 0 块 NVIDIA GPU
    OrtCUDAProviderOptions cudaOptions{};
    cudaOptions.device_id = 0;
    sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);

    // 模型路径
    const std::string modelPath = "/home/project/modle/bestv26s.onnx";

    // 加载 ONNX 模型，创建一个可以执行推理的对象session
    // env：之前创建的 ONNX Runtime 环境，负责日志等运行基础设施。
    // modelPath.c_str()：模型文件路径。
    // sessionOptions：推理配置，包括图优化和CUDA。
    Ort::Session session(env, modelPath.c_str(), sessionOptions);

    // 创建默认内存分配器，供 ONNX Runtime 分配和释放内存
    Ort::AllocatorWithDefaultOptions allocator;

    // 获取模型输入和输出的名字
    auto inputName = session.GetInputNameAllocated(0, allocator);
    auto outputName = session.GetOutputNameAllocated(0, allocator);

    const char *inputNames[] = {inputName.get()};
    const char *outputNames[] = {outputName.get()};

    // 与 CUDA Execution Provider 使用同一块 GPU。
    checkCuda(cudaSetDevice(0), "cudaSetDevice");

    const int inputWidth = 960;
    const int inputHeight = 960;

    // 置信度阈值
    const float confidenceThreshold = 0.20f;

    // 非极大值抑制阈值
    const float nmsThreshold = 0.45f;

    // 显存放在循环外，重复使用。
    // 放在采集线程创建之前，避免分配失败时留下未 join 的线程。
    GpuBuffers gpu;

    // 检查 CUDA 操作是否成功，失败时抛出异
    checkCuda(
        allocateInput(&gpu.input, inputWidth, inputHeight),
        "allocateInput");

    // 循环外分配一次，后续每帧重复使用。
    checkCuda(
        allocatePostprocess(&gpu.output),
        "allocatePostprocess");

    // 分配解析后的 5 个检测框。
    checkCuda(
        allocateDetections(&gpu.detections),
        "allocateDetections");

    checkCuda(
        allocateSelectedDetection(&gpu.selected),
        "allocateSelectedDetection");

    // 数据位置，在编号为 0 的 CUDA GPU 显存中
    Ort::MemoryInfo cudaMemoryInfo(
        "Cuda",             // 内存所在设备：CUDA GPU
        OrtDeviceAllocator, // 分配器类型：设备分配器
        0,                  // GPU 编号：第 0 块 GPU
        OrtMemTypeDefault   // 默认内存类型
    );

    std::array<int64_t, 4> inputShape = {
        1,           // 一次输入 1 张图片
        3,           // 每张图片有 3 个颜色通道
        inputHeight, // 输入高度
        inputWidth   // 输入宽度
    };

    // 输入图片的元素总数
    const size_t inputElementCount =
        static_cast<size_t>(3) * inputHeight * inputWidth;

    // 把已经存在的 gpu.input 显存包装成 ONNX Runtime 可以使用的输入张量
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        cudaMemoryInfo,
        gpu.input, // 数据所在的显存地址
        inputElementCount,
        inputShape.data(),  // 形状数组的地址，例如 {1, 3, 640, 640}
        inputShape.size()); // 维度数量，这里是 4

    // 固定输出形状，必须与模型实际输出一致。
    // gpu.output 已分配 30 个 float。
    std::array<int64_t, 3> gpuOutputShape = {1, 5, 6};

    // 把已经存在的 gpu.output 显存包装成输出张量
    Ort::Value outputTensor = Ort::Value::CreateTensor<float>(
        cudaMemoryInfo,
        gpu.output,
        30,
        gpuOutputShape.data(),
        gpuOutputShape.size());

    // 为推理会话 session 创建一个叫 ioBinding 的输入、输出绑定对象
    Ort::IoBinding ioBinding(session);

    ioBinding.BindInput(inputName.get(), inputTensor);
    ioBinding.BindOutput(outputName.get(), outputTensor);

    // 启动 RealSense 相机并获取一帧彩色图像
    // 配置 D435i 的彩色流和深度流
    // 创建管道是为了更好的调用SDK
    // 创建数据管道，负责管理相机的数据采集和帧传输
    rs2::pipeline pipeline;
    // 创建配置对象，用来指定需要开启哪些数据流，以及它们的分辨率、格式和帧率
    rs2::config config;

    // 配置彩色数据流，参数含义为彩色图像，宽，高，按 蓝、绿、红(BGR) 排列的三个通道，每个通道 8 位,与OpenCV 常用的颜色顺序一致。每秒频率为30帧
    // enable_stream为启动数据流函数
    config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 60);

    // 配置深度数据流,参数为深度图像,宽,高,每个通道16位,每秒频率30帧
    config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 60);

    // 打开摄像头，应用 config 中设置的参数
    auto profile = pipeline.start(config);

    // 获取 RGB 彩色相机
    auto color_sensor = profile.get_device().first<rs2::color_sensor>();

    // 自动曝光
    color_sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1.f);

    // 定义一个名为 frameMutex的互斥锁，用于协调采集线程和识别线程对共享数据的访问
    std::mutex frameMutex;

    // 创建一个变量，用于等待某个线程条件是否成立
    std::condition_variable frameCv;

    // 保存最近收到的一组相机帧
    rs2::frameset latestFrames;
    bool hasNewFrame = false;
    bool stopCapture = false;
    std::exception_ptr captureError;

    // 保存上一帧数据，并跳过第一次循环
    yuce::Coordinate resultCopy{};
    bool hasresultCopy = false;

    // 保存各帧的速度数据，用于计算平均加速度
    std::vector<yuce::FrameVelocity> velocityData;
    int frameIndex = 0; // 获取 CPU 内存中的图片地址

    // 共享计数，所有读写都用 frameMutex 保护
    std::uint64_t receivedCount = 0;
    std::uint64_t processedCount = 0;
    std::uint64_t overwrittenCount = 0;

    // 创建线程captureThread
    std::thread captureThread([&]()
                              {
    try {
        while (true) {
            {
                std::lock_guard<std::mutex> lock(frameMutex);
                if (stopCapture) {
                    break;
                }
            }

            rs2::frameset incoming;

            // 超时后重新检查退出标志，避免退出时一直等待相机
            if (!pipeline.try_wait_for_frames(&incoming, 100)) {
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(frameMutex);

                if (stopCapture) {
                    break;

                }

                // 覆盖旧帧：只保存最新的一组彩色和深度帧
                latestFrames = incoming;
                hasNewFrame = true;
            }

            frameCv.notify_one();
        }
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            captureError = std::current_exception();
        }
        frameCv.notify_one();
    } });

    auto stopCaptureThread = [&]()
    {
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            stopCapture = true;
        }

        frameCv.notify_all();

        if (captureThread.joinable())
        {
            captureThread.join();
        }
    };

    // 将深度图对齐到彩色图
    rs2::align alignToColor(RS2_STREAM_COLOR);

    try
    {

        while (true)
        {
            rs2::frameset frames;
            {
                std::unique_lock<std::mutex> lock(frameMutex);
                frameCv.wait(lock, [&]()
                             { return hasNewFrame || captureError; });

                if (captureError)
                {
                    std::rethrow_exception(captureError);
                }

                frames = latestFrames;
                hasNewFrame = false;
            }

            // 将深度图对齐到彩色图
            rs2::frameset alignedFrames = alignToColor.process(frames);

            // 获取彩色帧和深度帧
            rs2::video_frame colorFrame = alignedFrames.get_color_frame();
            rs2::depth_frame depthFrame = alignedFrames.get_depth_frame();

            if (!colorFrame || !depthFrame)
            {
                std::cerr << "无法获取彩色图像\n";
                continue;
            }

            // 彩色帧的像素数据包装成 OpenCV 的图像对象 image
            cv::Mat image(
                colorFrame.get_height(),
                colorFrame.get_width(),
                CV_8UC3,
                // 像素数据地址
                const_cast<void *>(colorFrame.get_data()),
                // 每行数据占用的字节数，包含可能的填充
                colorFrame.get_stride_in_bytes());

            image = image.clone();

            // 原图尺寸没有变化时，不会重复分配显存。
            gpu.ensureImageSize(
                colorFrame.get_width(),
                colorFrame.get_height());

            // 上传原始 BGR 图像。
            // 使用 RealSense 实际的行跨度，兼容行尾填充。
            checkCuda(
                uploadImage(
                    gpu.image,
                    static_cast<const unsigned char *>(colorFrame.get_data()),
                    colorFrame.get_width(),
                    colorFrame.get_height(),
                    static_cast<size_t>(colorFrame.get_stride_in_bytes())),
                "uploadImage");

            float scale = 1.0f; // 图像等比例缩放的比例
            int padLeft = 0;    // 缩放后图像左侧的补边像素数
            int padTop = 0;     // 缩放后图像顶部的补边像素数

            // GPU 完成缩放、补边、BGR→RGB、归一化、HWC （高度 宽度 通道数） →CHW
            checkCuda(
                preprocessImage(
                    gpu.image,
                    colorFrame.get_width(),
                    colorFrame.get_height(),
                    gpu.input,
                    inputWidth,
                    inputHeight,
                    scale,
                    padLeft,
                    padTop),
                "preprocessImage");

            // 等待 GPU 预处理完成。
            checkCuda(
                cudaDeviceSynchronize(),
                "synchronize preprocessing");

            // 使用已绑定的输入和输出张量执行一次模型推理
            session.Run(Ort::RunOptions{nullptr}, ioBinding);

            // 等待绑定的输出数据就绪，供后续 GPU 后处理使用
            ioBinding.SynchronizeOutputs();

            // 第一步：GPU 解析检测框。
            checkCuda(
                parseDetections(
                    gpu.output,
                    gpu.detections),
                "parseDetections");

            // 第二步：GPU 过滤无效框。
            checkCuda(
                filterDetections(
                    gpu.detections,
                    confidenceThreshold),
                "filterDetections");

            // GPU：选择最高分目标，写入 gpu.selected。
            checkCuda(
                selectTarget(gpu.detections, gpu.selected),
                "selectTarget");

            // GPU：还原到原图坐标。
            checkCuda(
                restoreTargetCoordinates(
                    gpu.selected,
                    scale,
                    padLeft,
                    padTop,
                    image.cols,
                    image.rows),
                "restoreTargetCoordinates");

            // 两个核函数使用同一个默认 stream，按顺序执行。
            // 最后统一等待解析和过滤完成。
            checkCuda(
                // cudaDeviceSynchronize() 让 CPU 等待当前 GPU 上此前提交的任务全部完成，并返回执行状态。
                cudaDeviceSynchronize(),
                "synchronize postprocessing");

            // 调用 .cu 中的函数，将最终目标拷回 CPU。
            Detection selected{};

            checkCuda(
                copySelectedToCPU(
                    gpu.selected,
                    &selected),
                "copySelectedToCPU");

            if (selected.valid)
            {
                const float x1 = selected.x1;
                const float y1 = selected.y1;
                const float x2 = selected.x2;
                const float y2 = selected.y2;

                const float bestScore = selected.score;
                const int classId = selected.classId;
                // 只绘制与原图有交集的有效框
                // floor()为向下取整函数,ceil()为向上取证函数
                if (x2 > x1 && y2 > y1)
                {
                    const int left = static_cast<int>(std::floor(x1));
                    const int top = static_cast<int>(std::floor(y1));
                    const int right = static_cast<int>(std::ceil(x2));
                    const int bottom = static_cast<int>(std::ceil(y2));

                    cv::Rect box(left, top, right - left, bottom - top);

                    // 绘制绿色检测框
                    cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);

                    // 显示类别和置信度
                    std::string label = cv::format("class %d: %.2f", classId, bestScore);

                    const float originalCenterX = selected.centerX;
                    const float originalCenterY = selected.centerY;

                    // 默认没有有效深度
                    std::string depthLabel = " depth: N/A";

                    // 先检查浮点中心坐标是否在深度图范围内
                    // isfinite（）判断是否为有限值
                    if (std::isfinite(originalCenterX) &&
                        std::isfinite(originalCenterY) &&
                        originalCenterX >= 0.0f &&
                        originalCenterX < depthFrame.get_width() &&
                        originalCenterY >= 0.0f &&
                        originalCenterY < depthFrame.get_height())
                    {
                        // 四舍五入到最近的像素，并处理右侧、下侧边缘的取整情况
                        const int pixelX = std::min(
                            static_cast<int>(std::round(originalCenterX)),
                            depthFrame.get_width() - 1);

                        const int pixelY = std::min(
                            static_cast<int>(std::round(originalCenterY)),
                            depthFrame.get_height() - 1);

                        // 读取该像素的深度，单位为米
                        const float depthMeters = depthFrame.get_distance(pixelX, pixelY);

                        if (std::isfinite(depthMeters) && depthMeters > 0.0f)
                        {
                            depthLabel = cv::format(" depth: %.3f m", depthMeters);
                        }
                        else
                        {
                            std::cout << "预测中心处没有有效深度\n";
                        }

                        // 获取对齐后深度帧对应的内参
                        const auto intrinsics = depthFrame.get_profile()
                                                    .as<rs2::video_stream_profile>()
                                                    .get_intrinsics();

                        const float pixel[2] = {
                            static_cast<float>(pixelX),
                            static_cast<float>(pixelY)};

                        float rawPoint[3] = {};

                        rs2_deproject_pixel_to_point(rawPoint, &intrinsics, pixel, depthMeters);

                        // 从米转换为毫米，存入坐标结构
                        yuce::Coordinate point{};
                        point.x = rawPoint[0] * 1000.0f;
                        point.y = rawPoint[1] * 1000.0f;
                        point.z = rawPoint[2] * 1000.0f;

                        // std::cout << "相机内部坐标：(" << point.x << ", " << point.y << "，" << point.z << ")\n";
                        yuce::Coordinate result = yuce::transformCoordinates(point.x, point.y, point.z, -0.3796);
                        // std::cout << "现实空间坐标：(" << result.x << ", " << result.y << "，" << result.z << ")\n";

                        if (hasresultCopy)
                        {
                            // 计算速度
                            yuce::Coordinate velocity = yuce::calculateVelocity(resultCopy, result, 16.67);
                            std::cout << "现实空间速度mm/ms：(" << velocity.x << ", " << velocity.y << "，" << velocity.z << ")\n";

                            // 保存当前帧速度
                            velocityData.push_back({frameIndex, velocity});

                            // 收集20帧速度后，计算这段时间的平均加速度
                            if (velocityData.size() == 30)
                            {
                                const yuce::FrameAcceleration acceleration = yuce::calculateAcceleration(velocityData, 16.67);

                                std::cout << "平均加速度mm/ms²：(" << acceleration.ax << ", " << 1000 * acceleration.ay << ", " << acceleration.az << ")\n";

                                if (acceleration.ay * 1000 >= 9)
                                {

                                    std::cout << "yes\n";
                                }

                                // 保留19帧，下一帧到来后继续更新
                                velocityData.erase(velocityData.begin());
                            }
                        }

                        // 计算完速度后保存，供下一帧使用
                        resultCopy = result;
                        hasresultCopy = true;
                    }

                    // 把深度文字追加到 label 后面
                    label += depthLabel;

                    // 必须放在深度处理之后，才能显示追加后的文字
                    cv::putText(image, label,
                                cv::Point(left, std::max(top - 8, 20)),
                                cv::FONT_HERSHEY_SIMPLEX,
                                0.6, cv::Scalar(0, 255, 0), 2);
                }
            }
            else
            {
                std::cout << "没有达到置信度阈值的目标\n";
            }

            // 显示当前帧的识别结果，按esc退出
            cv::imshow("YOLO Detection", image);
            if (cv::waitKey(1) == 27)
            {
                break;
            }
        }
    }
    catch (...)
    {
        stopCaptureThread();
        pipeline.stop();
        cv::destroyAllWindows();
        throw;
    }

    stopCaptureThread();
    pipeline.stop();
    cv::destroyAllWindows();
    return 0;
}

int main(int argc, char **argv)
{
    try
    {
        return runApplication(argc, argv);
    }
    catch (const std::exception &error)
    {
        std::cerr << "程序启动或运行失败：" << error.what() << '\n';
        return 1;
    }
}
