#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <exception>

int main(){

    //配置 D435i 的彩色流和深度流
    //创建管道是为了更好的调用SDK
    //创建数据管道，负责管理相机的数据采集和帧传输
    rs2::pipeline pipeline;
    //创建配置对象，用来指定需要开启哪些数据流，以及它们的分辨率、格式和帧率
    rs2::config config;

    //配置彩色数据流，参数含义为彩色图像，宽，高，按 蓝、绿、红(BGR) 排列的三个通道，每个通道 8 位,与OpenCV 常用的颜色顺序一致。每秒频率为30帧
    //enable_stream为启动数据流函数
    config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);

    //配置深度数据流,参数为深度图像,宽,高,每个通道16位,每秒频率30帧
    config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    //定义一个布尔变量 started,初始值为 false,通常用来记录管道是否已经启动。
    bool started = false;

    //定义一个整形变量  ,程序正常关闭为0,否则报错
    int exit_code = 0;

    try{
        //启动相机，获取配置
        rs2::pipeline_profile profile = pipeline.start(config);
        started = true;
        std::cout << "摄像头已打开\n";

        // 原始深度值每增加 1,对应多少米
        //sensor的值为深度传感器内的函数值
        rs2::depth_sensor sensor =profile.get_device().first<rs2::depth_sensor>();

        //depth_scale的数值为sensor里的函数值按照get_depth_scale()中的函数转为的距离m
        float depth_scale = sensor.get_depth_scale();

        //创建对齐对象
        rs2::align align_to_color(RS2_STREAM_COLOR);

        while(true){

            //等待相机返回一组帧
            rs2::frameset frames = pipeline.wait_for_frames();

            //执行对齐
            //process函数为 RealSense SDK 为 align 对象提供的处理方法，负责执行图像对齐
            rs2::frameset align_frames = align_to_color.process(frames);

            //get_color_frame() 是 RealSense SDK 提供的方法，用于从一组帧中取出彩色图像帧
            rs2::video_frame color_frame =align_frames.get_color_frame();

            //get_depth_frame() 是 RealSense SDK 提供的方法，用于从一组帧中取出深度图像帧
            rs2::depth_frame depth_frame =align_frames.get_depth_frame();

            if (!color_frame || !depth_frame) {
                  continue;
              }

            //用 cv::Mat 包装 SDK 的像素数据，不复制数据。
            //CV_8UC3:8位无符号整数,3个通道。
            //使用实际行跨度，兼容每行可能存在的填充字节。
            cv::Mat color(
                color_frame.get_height(),
                color_frame.get_width(),
                CV_8UC3,
                const_cast<void*>(color_frame.get_data()),
                color_frame.get_stride_in_bytes()
            );

            cv::Mat depth(
                depth_frame.get_height(),
                depth_frame.get_width(),
                CV_16UC1,
                const_cast<void*>(depth_frame.get_data()),
                depth_frame.get_stride_in_bytes()
            );

            //把整张深度图中的原始深度值转换成毫米
            //convertTo() 是 OpenCV 中用于转换矩阵数据类型，并可同时对数值进行缩放和偏移的函数。
            cv::Mat depth_mm;
            depth.convertTo(
                depth_mm,
                CV_32F,
                //先将 depth_scale 转换成 double 类型，再乘以 1000
                // static_cast 是 C++ 的一种显式类型转换的语法
                static_cast<double>(depth_scale) * 1000.0 
            );

            // 获取深度图的高度和宽度
            int height = depth.rows;
            int width = depth.cols;

            float center_depth = depth_mm.at<float>(height / 2, width / 2);

            if (center_depth > 0.0f) {
                std::cout << "中心深度："
                //setprecision 是 C++ 中用来设置浮点数输出精度的操纵符,setprecision(1)为保留一位小数
                << std::fixed << std::setprecision(1)
                << center_depth << " mm\n";
            }else{
                std::cout << "中心像素暂无有效深度\n";
            }

            // 将 0~4000 mm 映射为灰度值 0~255。
            // convertTo 转成 uint8 时会饱和处理，限制在 0~255。
            cv::Mat depth8U;
            depth_mm.convertTo(depth8U, CV_8UC1, 255.0 / 4000.0);

            cv::imshow("color", color);
            cv::imshow("depth", depth8U);

            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 'Q') {
                break;
            }

        }

    }
    catch (const rs2::error& e) {
        std::cerr << "RealSense 错误："
                //e.get_failed_function()：获取发生错误的 RealSense API 函数名。
                //e.what()：获取具体错误描述
                << e.get_failed_function() << ":"
                << e.what() << '\n';
            exit_code = 1;
      } catch (const std::exception& e) {
          std::cerr << "程序错误：" << e.what() << '\n';
          exit_code = 1;
      }

    // 对应 Python finally 中的资源清理；
    // 即使停止相机失败，也继续关闭窗口。
    if (started) {
        try {
            pipeline.stop();
        } 
        catch (const std::exception& e) {
        std::cerr << "停止相机失败：" << e.what() << '\n';
        exit_code = 1;
        }
    }

    cv::destroyAllWindows();
    return exit_code;




}