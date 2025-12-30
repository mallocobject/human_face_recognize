cmake_minimum_required(VERSION 3.25)

project(human_face_recognize)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(/usr/include/eigen3/)

aux_source_directory(. SRC)

add_executable(HFR main.cpp ${SRC})

target_link_libraries(HFR ${OpenCV_LIBS})