#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

namespace color_pipeline {
    int start_pipeline_colors(std::vector<cv::Mat> color_images, std::vector<cv::Mat> resized_images);
}
