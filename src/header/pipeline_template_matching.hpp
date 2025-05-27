#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "../header/bounding_box.hpp"



namespace template_pipeline {
    std::vector<BoundingBox> start_pipeline_template_matching(std::vector<cv::Mat> shape_images, std::vector<std::vector<cv::Mat>> templates);
}