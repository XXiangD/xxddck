#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <librealsense2/rs.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>


int main() {
    //加载模型
    //cv::dnn::Net 是 OpenCV 中表示深度神经网络的类。dnn：深度神经网络模块的命名空间。Net：神经网络类
    const std::string modelPath = ("/home/humble/volleyball1/runs/detect/volleyball_m_project/5060_high_res-4/weights/best.onnx");

    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
    
    // 使用 OpenCV 自带的 DNN 推理后端
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);

    //使用CPU推理
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    //启动 RealSense 相机并获取一帧彩色图像
    //配置 D435i 的彩色流和深度流
    //创建管道是为了更好的调用SDK
    //创建数据管道，负责管理相机的数据采集和帧传输
    rs2::pipeline pipeline;
    //创建配置对象，用来指定需要开启哪些数据流，以及它们的分辨率、格式和帧率
    rs2::config config;

    //配置彩色数据流，参数含义为彩色图像，宽，高，按 蓝、绿、红(BGR) 排列的三个通道，每个通道 8 位,与OpenCV 常用的颜色顺序一致。每秒频率为30帧
    //enable_stream为启动数据流函数
    config.enable_stream(RS2_STREAM_COLOR,640, 480,RS2_FORMAT_BGR8,30);

    //配置深度数据流,参数为深度图像,宽,高,每个通道16位,每秒频率30帧
    config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    //打开摄像头，应用 config 中设置的参数
    pipeline.start(config);

    // 将深度图对齐到彩色图
    rs2::align alignToColor(RS2_STREAM_COLOR);

    while(true){
        // 等待并获取一组相机帧
    //wait_for_frames() 是 RealSense SDK 中用于等待并获取一组新帧的函数
    rs2::frameset frames = pipeline.wait_for_frames();

    // 将深度图对齐到彩色图
    rs2::frameset alignedFrames = alignToColor.process(frames);

    //获取彩色帧和深度帧
    rs2::video_frame colorFrame = frames.get_color_frame();
    rs2::depth_frame depthFrame = alignedFrames.get_depth_frame();

    if (!colorFrame || !depthFrame) {
        std::cerr << "无法获取彩色图像\n";
        continue;
    }

    //彩色帧的像素数据包装成 OpenCV 的图像对象 image
    cv::Mat image(
        colorFrame.get_height(),
        colorFrame.get_width(),
        CV_8UC3,
        // 像素数据地址
        const_cast<void*>(colorFrame.get_data()),
        //每行数据占用的字节数，包含可能的填充
        colorFrame.get_stride_in_bytes()
    );

    image = image.clone();

    //定义不可改变变量宽，高，置信度阈值
    const int inputWidth = 640;
    const int inputHeight = 640;
    const float confidenceThreshold = 0.50f;

    //等比例缩放并补边，避免把 640×480 直接拉伸为正方形
    const float scale = std::min(
        inputWidth / static_cast<float>(image.cols),
        inputHeight / static_cast<float>(image.rows)
    );

    const int resizedWidth = static_cast<int>(
        std::round(image.cols * scale)
    );
    const int resizedHeight = static_cast<int>(
        std::round(image.rows * scale)
    );

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resizedWidth, resizedHeight));

    //计算补充画布
    const int padLeft = (inputWidth - resizedWidth) / 2;
    const int padTop = (inputHeight - resizedHeight) / 2;

    //完整画布
    cv::Mat padded(
          inputHeight, inputWidth, CV_8UC3,
          cv::Scalar(114, 114, 114)
    );

    //把缩放后的图像 resized，复制到画布 padded 的指定区域
    resized.copyTo(
          padded(cv::Rect(padLeft, padTop, resizedWidth, resizedHeight))
    );

    //转为模型输入：
    // 像素除以 255；BGR 转 RGB；生成 [1, 3, 640, 640] 的张量
    //cv::dnn::blobFromImage() 用于把 OpenCV 图像转换为神经网络需要的输入张量
    //cv::Scalar() 创建一个各分量都为 0 的标量对象
    cv::Mat blob = cv::dnn::blobFromImage(
        padded,
        1.0 / 255.0,
        cv::Size(inputWidth, inputHeight),
        cv::Scalar(),   // 各通道减去的均值，这里为 0
        true,           // swapRB：BGR 转 RGB
        false           // 不裁剪
    );

    //把数据交给模型
    net.setInput(blob);

    //执行推理
    //创建动态数组，存放模型输出的张量
    std::vector<cv::Mat> outputs;

    //神经网络执行推理，并把预测结果保存到 outputs
    //forward() 表示执行神经网络的前向推理
    //net.getUnconnectedOutLayersNames() 用于获取网络中没有连接到后续层的那些层的名称，通常就是模型的最终输出层
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    cv::Mat output = outputs[0];

    //整理成每行一个候选框：
    // cx、cy、w、h、类别分数
    cv::Mat predictions;

    if (output.size[1] == 5 && output.size[2] != 5) {
        // 把模型输出的数据按二维矩阵查看
        cv::Mat matrix(
            5,                    // 行数：5 项属性
            output.size[2],       // 列数：N 个候选框
            CV_32F,               // 每个元素是单通道 32 位浮点数
            output.ptr<float>()   // 模型输出数据的起始地址
        );

    //转置矩阵
    // predictions 存放整理后的模型预测结果
    cv::transpose(matrix, predictions);

    } else if (output.size[2] == 5 && output.size[1] != 5) {
        // [1, N, 5] → [N, 5]，无需转置
        predictions = cv::Mat(
            output.size[1],      // 行数：N 个候选框
            5,                   // 列数：每个框的 5 项属性
            CV_32F,              // 元素类型：单通道 32 位浮点数
            output.ptr<float>()  // 输出数据的起始地址
        );


    } else {
        std::cerr << "输出形状不支持，或 [1, 5, 5] 的排列方向存在歧义\n";
        return 1;
    }

    //记录分数最高且达到置信度阈值的候选框编号
    int bestIndex = -1;
    //记录当前最高分数
    float bestScore = -1.0f;

    for (int i = 0; i < predictions.rows; ++i) {
        const float* data = predictions.ptr<float>(i);

        const float centerX = data[0];
        const float centerY = data[1];
        const float width   = data[2];
        const float height  = data[3];
        const float score   = data[4];

        // 跳过无效数据
        if (!std::isfinite(centerX) ||
            !std::isfinite(centerY) ||
            !std::isfinite(width) ||
            !std::isfinite(height) ||
            !std::isfinite(score) ||
            width <= 0.0f ||
            height <= 0.0f) {
            continue;
        }

        if (score >= confidenceThreshold && score > bestScore) {
            bestScore = score;
            bestIndex = i;
        }
    }

    //读取选中目标的坐标
    if (bestIndex >= 0) {
        const float* target = predictions.ptr<float>(bestIndex);

        const float centerX = target[0];
        const float centerY = target[1];
        const float width   = target[2];
        const float height  = target[3];
        const int classId   = 0;  // 只有一个类别

        std::cout << "类别编号：" << classId << '\n';
        std::cout << "最高分数：" << bestScore << '\n';
        std::cout << "中心坐标：("
                  << centerX << ", " << centerY << ")\n";
        std::cout << "宽高：("
                  << width << ", " << height << ")\n";

        //模型输出的中心坐标和宽高即原图上的左上角、右下角
        float x1 = (centerX - width  / 2.0f - padLeft) / scale;
        float y1 = (centerY - height / 2.0f - padTop)  / scale;
        float x2 = (centerX + width  / 2.0f - padLeft) / scale;
        float y2 = (centerY + height / 2.0f - padTop)  / scale;

        // 将边界限制在原图范围内
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(image.cols)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(image.rows)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(image.cols)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(image.rows)));

        //只绘制与原图有交集的有效框
        //floor()为向下取整函数,ceil()为向上取证函数
        if (x2 > x1 && y2 > y1) {
            const int left   = static_cast<int>(std::floor(x1));
            const int top    = static_cast<int>(std::floor(y1));
            const int right  = static_cast<int>(std::ceil(x2));
            const int bottom = static_cast<int>(std::ceil(y2));

            cv::Rect box(left, top, right - left, bottom - top);

            // 绘制绿色检测框
            cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);

            // 显示类别和置信度
            std::string label =cv::format("class %d: %.2f", classId, bestScore);

            // 模型预测中心还原到原图，保留浮点精度
            const float originalCenterX = (centerX - padLeft) / scale;
            const float originalCenterY = (centerY - padTop) / scale;

            // 默认没有有效深度
            std::string depthLabel = " depth: N/A";

            // 先检查浮点中心坐标是否在深度图范围内
            //isfinite（）判断是否为有限值
            if (std::isfinite(originalCenterX) &&
                std::isfinite(originalCenterY) &&
                originalCenterX >= 0.0f &&
                originalCenterX < depthFrame.get_width() &&
                originalCenterY >= 0.0f &&
                originalCenterY < depthFrame.get_height()) {
                    // 四舍五入到最近的像素，并处理右侧、下侧边缘的取整情况
                    const int pixelX = std::min(
                        static_cast<int>(std::round(originalCenterX)),
                        depthFrame.get_width() - 1
                    );

                    const int pixelY = std::min(
                        static_cast<int>(std::round(originalCenterY)),
                        depthFrame.get_height() - 1
                    );

                    // 读取该像素的深度，单位为米
                    const float depthMeters = depthFrame.get_distance(pixelX, pixelY);
                    if (std::isfinite(depthMeters) && depthMeters > 0.0f) {
                        depthLabel = cv::format(" depth: %.3f m", depthMeters);
                        std::cout << "预测中心像素：("<< pixelX << ", " << pixelY << ")\n";

                        std::cout << "预测中心深度："<< depthMeters << " 米\n";
                    } else {
            std::cout << "预测中心处没有有效深度\n";
            }
        }
        // 把深度文字追加到 label 后面
        label += depthLabel;

        // 必须放在深度处理之后，才能显示追加后的文字
        cv::putText(image,label,
                    cv::Point(left, std::max(top - 8, 20)),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,cv::Scalar(0, 255, 0),2);

        } else {
        std::cout << "没有达到置信度阈值的目标\n";
    }
}
    //显示当前帧的识别结果，按esc退出
    cv::imshow("YOLO Detection", image);
    if (cv::waitKey(1) == 27) {
        break;
    }
        
    
}
pipeline.stop();
cv::destroyAllWindows();
return 0;

}
    

