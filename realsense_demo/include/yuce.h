//头文件保护
#ifndef YUCE_H
#define YUCE_H

#include <cmath>
#include <stdexcept>

namespace yuce {

struct Coordinate {
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
        y * sinPhi + z * cosPhi
    };
}

// 计算现实坐标系中的三个速度分量。
// previous：上一帧转换后的坐标。
// current：当前帧转换后的坐标。
// deltaTime：两帧时间间隔，单位为秒，必须大于 0。
// 坐标单位为米时，返回速度单位为米/秒。
// yuce::Coordinate velocity = yuce::calculateVelocity(previous, current, deltaTime);
inline Coordinate calculateVelocity(
    const Coordinate& previous,
    const Coordinate& current,
    double deltaTime)
{
    if (!std::isfinite(deltaTime) || deltaTime <= 0.0) {
        throw std::invalid_argument("时间错误");
    }

    return {
        (current.x - previous.x) / deltaTime,
        (current.y - previous.y) / deltaTime,
        (current.z - previous.z) / deltaTime
    };
}

 


} 

#endif 