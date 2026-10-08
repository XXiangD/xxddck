#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <librealsense2/rs.hpp>
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
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <cmath>

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
    const std::string modelPath = "/home/humble/volleyball1/runs/detect/volleyball_m_project/5060_high_res-4/weights/best.onnx";

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

    // 输入数据放在 CPU 内存中，不代表模型仍然使用 CPU 推理。启用 CUDA 后，ONNX Runtime 会处理需要的数据传输。
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,
        OrtMemTypeDefault);

    // 启动 RealSense 相机并获取一帧彩色图像
    // 配置 D435i 的彩色流和深度流
    // 创建管道是为了更好的调用SDK
    // 创建数据管道，负责管理相机的数据采集和帧传输
    rs2::pipeline pipeline;
    // 创建配置对象，用来指定需要开启哪些数据流，以及它们的分辨率、格式和帧率
    rs2::config config;

    // 配置彩色数据流，参数含义为彩色图像，宽，高，按 蓝、绿、红(BGR) 排列的三个通道，每个通道 8 位,与OpenCV 常用的颜色顺序一致。每秒频率为30帧
    // enable_stream为启动数据流函数
    config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);

    // 配置深度数据流,参数为深度图像,宽,高,每个通道16位,每秒频率30帧
    config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    // 打开摄像头，应用 config 中设置的参数
    pipeline.start(config);

    // 定义一个名为 frameMutex的互斥锁，用于协调采集线程和识别线程对共享数据的访问
    std::mutex frameMutex;

    // 创建一个变量，用于等待某个线程条件是否成立
    std::condition_variable frameCv;

    // 保存最近收到的一组相机帧
    rs2::frameset latestFrames;
    bool hasNewFrame = false;
    bool stopCapture = false;
    std::exception_ptr captureError;

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

            // 定义不可改变变量宽，高，置信度阈值
            const int inputWidth = 960;
            const int inputHeight = 960;
            const float confidenceThreshold = 0.50f;

            // 等比例缩放并补边，避免把 640×480 直接拉伸为正方形
            const float scale = std::min(
                inputWidth / static_cast<float>(image.cols),
                inputHeight / static_cast<float>(image.rows));

            const int resizedWidth = static_cast<int>(
                std::round(image.cols * scale));
            const int resizedHeight = static_cast<int>(
                std::round(image.rows * scale));

            cv::Mat resized;
            cv::resize(image, resized, cv::Size(resizedWidth, resizedHeight));

            // 计算补充画布
            const int padLeft = (inputWidth - resizedWidth) / 2;
            const int padTop = (inputHeight - resizedHeight) / 2;

            // 完整画布
            cv::Mat padded(
                inputHeight, inputWidth, CV_8UC3,
                cv::Scalar(114, 114, 114));

            // 把缩放后的图像 resized，复制到画布 padded 的指定区域
            resized.copyTo(
                padded(cv::Rect(padLeft, padTop, resizedWidth, resizedHeight)));

            // 转为模型输入：
            //  像素除以 255；BGR 转 RGB；生成 [1, 3, 640, 640] 的张量
            // cv::dnn::blobFromImage() 用于把 OpenCV 图像转换为神经网络需要的输入张量
            // cv::Scalar() 创建一个各分量都为 0 的标量对象
            cv::Mat blob = cv::dnn::blobFromImage(
                padded,
                1.0 / 255.0,
                cv::Size(inputWidth, inputHeight),
                cv::Scalar(), // 各通道减去的均值，这里为 0
                true,         // swapRB：BGR 转 RGB
                false         // 不裁剪
            );

            // blob 已经是 float32，形状为 [1, 3, 960, 960]
            // blob.isContinuous()：检查 OpenCV 的 cv::Mat 数据是否连续存放
            if (!blob.isContinuous())
            {
                blob = blob.clone();
            }

            std::array<int64_t, 4> inputShape = {1, 3, inputHeight, inputWidth};

            // 将 OpenCV 的 blob 内存包装为 ONNX Runtime 输入
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo,
                blob.ptr<float>(),
                blob.total(),
                inputShape.data(),
                inputShape.size());

            // 执行推理
            auto ortOutputs = session.Run(
                Ort::RunOptions{nullptr}, // 使用默认运行选项
                inputNames,
                &inputTensor,
                1,
                outputNames,
                1);

            // 你的模型已测得输出为 [1, 5, 18900]
            auto outputInfo = ortOutputs[0].GetTensorTypeAndShapeInfo();
            auto outputShape = outputInfo.GetShape();

            int sizes[] = {1, 5, 18900};

            cv::Mat outputView(
                3,
                sizes,
                CV_32F,
                ortOutputs[0].GetTensorMutableData<float>());

            // 当前循环内 ortOutputs 仍然有效，可以直接使用
            cv::Mat output = outputView;

            // 整理成每行一个候选框：
            //  cx、cy、w、h、类别分数
            cv::Mat predictions;

            if (output.size[1] == 5 && output.size[2] != 5)
            {
                // 把模型输出的数据按二维矩阵查看
                cv::Mat matrix(
                    5,                  // 行数：5 项属性
                    output.size[2],     // 列数：N 个候选框
                    CV_32F,             // 每个元素是单通道 32 位浮点数
                    output.ptr<float>() // 模型输出数据的起始地址
                );

                // 转置矩阵
                //  predictions 存放整理后的模型预测结果
                cv::transpose(matrix, predictions);
            }
            else if (output.size[2] == 5 && output.size[1] != 5)
            {
                // [1, N, 5] → [N, 5]，无需转置
                predictions = cv::Mat(
                    output.size[1],     // 行数：N 个候选框
                    5,                  // 列数：每个框的 5 项属性
                    CV_32F,             // 元素类型：单通道 32 位浮点数
                    output.ptr<float>() // 输出数据的起始地址
                );
            }
            else
            {
                std::cerr << "输出形状不支持，或 [1, 5, 5] 的排列方向存在歧义\n";
                return 1;
            }

            // 记录分数最高且达到置信度阈值的候选框编号
            int bestIndex = -1;
            // 记录当前最高分数
            float bestScore = -1.0f;

            for (int i = 0; i < predictions.rows; ++i)
            {
                const float *data = predictions.ptr<float>(i);

                const float centerX = data[0];
                const float centerY = data[1];
                const float width = data[2];
                const float height = data[3];
                const float score = data[4];

                // 跳过无效数据
                if (!std::isfinite(centerX) ||
                    !std::isfinite(centerY) ||
                    !std::isfinite(width) ||
                    !std::isfinite(height) ||
                    !std::isfinite(score) ||
                    width <= 0.0f ||
                    height <= 0.0f)
                {
                    continue;
                }

                if (score >= confidenceThreshold && score > bestScore)
                {
                    bestScore = score;
                    bestIndex = i;
                }
            }

            // 读取选中目标的坐标
            if (bestIndex >= 0)
            {
                const float *target = predictions.ptr<float>(bestIndex);

                const float centerX = target[0];
                const float centerY = target[1];
                const float width = target[2];
                const float height = target[3];
                const int classId = 0; // 只有一个类别

                // std::cout << "类别编号：" << classId << '\n';
                // std::cout << "最高分数：" << bestScore << '\n';
                // std::cout << "中心坐标：("
                //           << centerX << ", " << centerY << ")\n";
                // std::cout << "宽高：("
                //           << width << ", " << height << ")\n";

                // 模型输出的中心坐标和宽高即原图上的左上角、右下角
                float x1 = (centerX - width / 2.0f - padLeft) / scale;
                float y1 = (centerY - height / 2.0f - padTop) / scale;
                float x2 = (centerX + width / 2.0f - padLeft) / scale;
                float y2 = (centerY + height / 2.0f - padTop) / scale;

                // 将边界限制在原图范围内
                x1 = std::max(0.0f, std::min(x1, static_cast<float>(image.cols)));
                y1 = std::max(0.0f, std::min(y1, static_cast<float>(image.rows)));
                x2 = std::max(0.0f, std::min(x2, static_cast<float>(image.cols)));
                y2 = std::max(0.0f, std::min(y2, static_cast<float>(image.rows)));

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

                    // 模型预测中心还原到原图，保留浮点精度
                    const float originalCenterX = (centerX - padLeft) / scale;
                    const float originalCenterY = (centerY - padTop) / scale;

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

                        float point[3] = {};

                        rs2_deproject_pixel_to_point(point, &intrinsics, pixel, depthMeters);

                        point[0] = 1000*point[0];
                        point[1] = 1000*point[1];
                        point[2] = 1000*point[2];

                        std::cout << "相机内部坐标：(" << point[0] << ", " << point[1] << "，" << point[2] << ")\n";
                        yuce::Coordinate result = yuce::transformCoordinates(point[0],point[1], point[2], 0.5236);
                        std::cout << "现实空间坐标：(" << result.x << ", " << result.y << "，" << result.z << ")\n";
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
