cmake_minimum_required(VERSION 3.10)
project(ImageProcessingCpp)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${FFTW_INCLUDE_DIRS})

set(SOURCES
        src/pipelines/preprocessing_pipeline.cpp
        src/pipelines/pipeline_colors.cpp
        src/pipelines/pipeline_shapes.cpp
        src/pipelines/main_pipeline.cpp
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
        src/pipelines/preprocessing_pipeline.cpp
        src/pipelines/pipeline_colors.cpp
        src/pipelines/pipeline_shapes.cpp
        src/pipelines/main_pipeline.cpp
        src/colors.cpp
        src/color_detection.cpp
        src/bounding_box.cpp
        src/basic_image_operations.cpp
        src/statistical_operations.cpp
        src/geometrical_image_operations.cpp
        src/filters.cpp
        src/shape_detection.cpp
        src/pipelines/pipeline_shapes.cpp
        src/header/pipeline_shapes.hpp
        src/pipelines/pipeline_box_fusion.cpp
        src/header/pipeline_box_fusion.hpp
        src/pipelines/main_pipeline.cpp
        src/header/main_pipeline.hpp
        src/header/preprocessing_pipeline.hpp)

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${FFTW_LIBRARIES}
        -lfftw3f)

target_include_directories(${PROJECT_NAME} PRIVATE /usr/include)




