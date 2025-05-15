#pragma once
#include <opencv2/opencv.hpp>
#include <string>

namespace geo_ops {
    cv::Mat resize_image(const cv::Mat& image, int target_width, int target_height);

    cv::Mat rotate_image(const cv::Mat& image, int degree);

    cv::Mat mirror_image(const cv::Mat& image, const std::string& mode = "vertical");
}
