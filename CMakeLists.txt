cmake_minimum_required(VERSION 3.10)
project(ImageProcessingCpp)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${FFTW_INCLUDE_DIRS})

set(SOURCES
        src/pipelines/preprocessing.cpp
        src/pipelines/pipeline_colors.cpp
        src/basic_image_operations.cpp
        src/statistical_operations.cpp
        src/geometrical_image_operations.cpp
        src/filters.cpp
        src/color_detection.cpp
        src/colors.cpp
        src/shape_detection.cpp
        src/bounding_box.cpp
        src/header/pipeline_colors.hpp
)

add_executable(${PROJECT_NAME}
        src/pipelines/preprocessing.cpp
        src/pipelines/pipeline_colors.cpp
        src/colors.cpp
        src/color_detection.cpp
        src/bounding_box.cpp
        src/basic_image_operations.cpp
        src/statistical_operations.cpp
        src/geometrical_image_operations.cpp
        src/filters.cpp
        src/shape_detection.cpp)

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${FFTW_LIBRARIES}
        -lfftw3f)

target_include_directories(${PROJECT_NAME} PRIVATE /usr/include)




