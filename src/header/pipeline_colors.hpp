#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "../header/bounding_box.hpp"

namespace color_pipeline {
    std::vector<BoundingBox> start_pipeline_colors(std::vector<cv::Mat> color_images);
}
