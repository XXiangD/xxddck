import cv2 
#裁剪图片、修改像素、计算检测框坐标和距离
import numpy as np
# D435i 需要用 RealSense SDK 的 Python 库 pyrealsense2 读取深度图
import pyrealsense2 as rs

# 配置 D435i 的彩色流和深度流
#创建管道是为了更好的调用SDK
#. 表示访问某个模块或对象里的成员
#创建数据管道，负责管理相机的数据采集和帧传输
pipeline = rs.pipeline()
#创建配置对象，用来指定需要开启哪些数据流，以及它们的分辨率、格式和帧率
config = rs.config()
#enable_stream为启动数据流函数
#配置彩色数据流，参数含义为彩色图像，宽，高，按 蓝、绿、红（BGR） 排列的三个通道，每个通道 8 位，与OpenCV 常用的颜色顺序一致。每秒频率为30帧
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#配置深度数据流，参数含义为深度图像，宽，高，每个像素以一个 16 位无符号整数存储深度值，每秒频率为30帧
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

#按照 config 中的设置启动相机，并把实际启动的配置信息保存到 profile
#start（）函数表示打开摄像头
#profile 保存的是相机管道启动后实际生效的配置描述，包括当前使用的相机设备
#config 保存了你希望相机使用的设置，启动时需要把这些设置交给管道
profile = pipeline.start(config)
print("摄像头已打开")

try:

    #profile 保存的是相机管道启动后实际生效的配置描述
    #get_device()为获取设备的函数，first_depth_sensor()为获取深度传感器的函数，get_depth_scale()为获取深度换算比例的函数
    #profile是管道启动后返回的“配置对象”，所以.get_device().first_depth_sensor().get_depth_scale()就不用在前面pipeline.，直接profile.就可以了
    #从配置对象找到相机，再找到深度传感器，最后读取深度换算比例
    #depth_scale 里面是一个浮点数，表示原始深度值每增加 1，对应多少米
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    #创建一个对齐处理对象，以彩色图像作为对齐目标
    #D435i 的彩色图和深度图来自不同位置的成像组件，观察角度和视野不同，若不对齐画面中的位置会有差别。
    align = rs.align(rs.stream.color)

    while True:

        #等待相机采集到一组新的数据帧，并让变量frames保存返回的各个画面
        #wait_for_frames()为等待并获取相机新采集的一组图像帧
        frames = pipeline.wait_for_frames()

        #把相机采集的帧交给对齐对象处理，再把处理结果赋给 aligned_frames
        #process函数为 RealSense SDK 为 align 对象提供的处理方法，负责执行图像对齐
        aligned_frames = align.process(frames)

        #get_color_frame() 是 RealSense SDK 提供的方法，用于从一组帧中取出彩色图像帧
        color_frame = aligned_frames.get_color_frame()

        #get_depth_frame() 是 RealSense SDK 提供的方法，用于从一组帧中取出深度图像帧
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        #取出彩色帧和深度帧中的像素数据，并转换成 NumPy 数组，方便后续计算和显示
        #np.asanyarray() 是 NumPy 提供的函数，用于把输入的数据转换为 NumPy 数组，方便按位置读取、计算和处理
        #get_data() 获取的是点号前面那个帧对象中的像素数据
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        #把整张深度图中的原始深度值转换成毫米
        #depth.astype(np.float32)将深度数组转换为 32 位浮点数类型，方便保存带小数的计算结果
        #astype()为转换数组中元素的数据类型
        depth_mm = depth.astype(np.float32) * depth_scale * 1000.0

        #获取深度图的高度和宽度，分别赋给两个变量
        #depth.shape 是数组的形状。对于二维深度图，它返回：行数（高度）, 列数(宽度)
        height, width = depth.shape

        #中心点深度
        center_depth = depth_mm[height // 2, width // 2]

        if center_depth > 0:
            print(f"中心深度：{center_depth:.1f} mm")
        else:
            print("中心像素暂无有效深度")

        # 将 0～4000 mm 映射到灰度值 0～255
        #np.clip() 是 NumPy 的函数，用于把第一个参数限制在第二三个参数范围内
        depth8U = np.clip(depth_mm * 255.0 / 4000.0, 0, 255).astype(np.uint8)

        cv2.imshow("color", color)
        cv2.imshow("depth", depth8U)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()



          

    