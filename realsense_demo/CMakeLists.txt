cmake_minimum_required(VERSION 3.16)
project(d435i_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui)

add_executable(d435i_demo main.cpp)

target_include_directories(d435i_demo PRIVATE
      ${OpenCV_INCLUDE_DIRS}
  )

target_link_libraries(d435i_demo PRIVATE
      realsense2::realsense2
      ${OpenCV_LIBS}
  )
