#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>

namespace sd {

    bool is_edge(const cv::Mat& binary_img, int y, int x);

    std::vector<cv::Point> trace_contour(const cv::Mat& binary_image, cv::Mat& visited, int y, int x);

    std::vector<std::vector<cv::Point>> get_contours(const cv::Mat& binary_image, int angle_tolerance = 10);

    std::tuple<cv::Point2f, float, float, float> min_area_rect(const std::vector<cv::Point>& contour);

    std::vector<cv::Point2f> rotate_points(const std::vector<cv::Point2f>& points, float angle);

    std::vector<cv::Point2f> get_rectangle_corners(const cv::Point2f& center, float width, float height, float angle);

}
