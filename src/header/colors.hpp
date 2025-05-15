#ifndef COLORS_HPP
#define COLORS_HPP

#include <opencv2/opencv.hpp>
#include <functional>

namespace colors {
    cv::Vec3f bgr_to_hsv(const cv::Vec3b& bgr);

    bool is_strong_red(float h, float s, float v);
    bool is_strong_green(float h, float s, float v);
    bool is_strong_blue(float h, float s, float v);
    bool is_strong_yellow(float h, float s, float v);

    cv::Mat get_mask(const cv::Mat& image, const std::function<bool(float, float, float)>& color_function);

    cv::Vec3b get_color_from_function(const std::function<bool(float, float, float)>& color_function);
}
#endif // COLORS_HPP

