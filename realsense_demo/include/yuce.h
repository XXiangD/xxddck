// 头文件保护
#ifndef YUCE_H
#define YUCE_H

#include <cmath>
#include <stdexcept>
#include <vector>

namespace yuce
{

    struct Coordinate
    {
        float x;
        float y;
        float z;
    };

    // 将相机坐标 (x, y, z) 转换为现实坐标 (x', y', z')。
    // phi：相机与水平面的夹角，单位为弧度。
    // 坐标原点保持在相机位置，现实坐标的 y' 轴向下。
    // yuce::Coordinate result = yuce::transformCoordinates(x, y, z, phi);
    inline Coordinate transformCoordinates(
        float x, float y, float z, float phi)
    {
        const float cosPhi = std::cos(phi);
        const float sinPhi = std::sin(phi);

        return {
            x,
            y * cosPhi - z * sinPhi,
            y * sinPhi + z * cosPhi};
    }

    // 计算现实坐标系中的三个速度分量。
    // previous：上一帧转换后的坐标。
    // current：当前帧转换后的坐标。
    // deltaTime：两帧时间间隔，单位为毫秒，必须大于 0。
    // 坐标单位为米时，返回速度单位为和毫米/毫秒。
    // yuce::Coordinate velocity = yuce::calculateVelocity(previous, current, deltaTime);
    inline Coordinate calculateVelocity(
        const Coordinate &previous,
        const Coordinate &current,
        double deltaTime)
    {
        if (!std::isfinite(deltaTime) || deltaTime <= 0.0)
        {
            throw std::invalid_argument("时间错误");
        }

        return {
            static_cast<float>((current.x - previous.x) / deltaTime),
            static_cast<float>((current.y - previous.y) / deltaTime),
            static_cast<float>((current.z - previous.z) / deltaTime)};
    }

    // 每帧的速度数据，速度单位：mm/ms
    struct FrameVelocity
    {
        int frameIndex;
        Coordinate velocity;
        double timeMs; // 速度对应的时间，单位ms
    };

    // 相邻两帧之间的加速度，单位：mm/ms²
    struct FrameAcceleration
    {
        int frameIndex; // 对应后一帧
        double ax;
        double ay;
        double az;
    };

    // 用30帧速度进行直线拟合，得到一组加速度。
    // 输入速度：mm/ms。
    // deltaTime：固定帧间隔，单位为毫秒。
    // 输出加速度：mm/ms²。
    // const yuce::FrameAcceleration acceleration =yuce::calculateAcceleration(velocityData, 16.67);
    inline FrameAcceleration calculateAcceleration(const std::vector<FrameVelocity> &frames, double deltaTime = 16.67)
    {
        if (frames.size() != 30)
        {
            throw std::invalid_argument("需要10帧速度数据");
        }

        if (!std::isfinite(deltaTime) || deltaTime <= 0.0)
        {
            throw std::invalid_argument("帧间隔必须为有限的正数");
        }

        // 第0帧时间为0，第29帧时间为19 * deltaTime
        const double meanTime = 14.5 * deltaTime;

        double numeratorX = 0.0;
        double numeratorY = 0.0;
        double numeratorZ = 0.0;
        double denominator = 0.0;

        for (std::size_t i = 0; i < frames.size(); ++i)
        {
            const double time = static_cast<double>(i) * deltaTime;
            const double centeredTime = time - meanTime;

            numeratorX += centeredTime * frames[i].velocity.x;
            numeratorY += centeredTime * frames[i].velocity.y;
            numeratorZ += centeredTime * frames[i].velocity.z;

            denominator += centeredTime * centeredTime;
        }

        return {
            frames.back().frameIndex,
            numeratorX / denominator,
            numeratorY / denominator,
            numeratorZ / denominator};
    }

}

#endif