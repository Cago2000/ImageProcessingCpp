#include "header/colors.hpp"

namespace colors {
    cv::Vec3b get_color_from_function(const std::function<bool(float, float, float)>& color_function) {
        static std::function<bool(float, float, float)> red_func = is_strong_red;
        static std::function<bool(float, float, float)> green_func = is_strong_green;
        static std::function<bool(float, float, float)> yellow_func = is_strong_yellow;
        static std::function<bool(float, float, float)> blue_func = is_strong_blue;

        auto color_ptr = color_function.target<bool(*)(float, float, float)>();
        auto red_ptr = red_func.target<bool(*)(float, float, float)>();
        auto green_ptr = green_func.target<bool(*)(float, float, float)>();
        auto yellow_ptr = yellow_func.target<bool(*)(float, float, float)>();
        auto blue_ptr = blue_func.target<bool(*)(float, float, float)>();

        if (color_ptr && red_ptr && *color_ptr == *red_ptr) {
            return cv::Vec3b(0, 0, 255); // Red
        }
        if (color_ptr && green_ptr && *color_ptr == *green_ptr) {
            return cv::Vec3b(0, 255, 0); // Green
        }
        if (color_ptr && yellow_ptr && *color_ptr == *yellow_ptr) {
            return cv::Vec3b(0, 255, 255); // Yellow
        }
        if (color_ptr && blue_ptr && *color_ptr == *blue_ptr) {
            return cv::Vec3b(255, 0, 0); // Blue
        }

        return cv::Vec3b(0, 0, 0); // Default black if unknown
    }

    cv::Vec3f bgr_to_hsv(const cv::Vec3b& bgr_pixel) {
        cv::Mat3b bgr(1, 1, bgr_pixel);
        cv::Mat3b hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        cv::Vec3b hsv_pixel = hsv.at<cv::Vec3b>(0, 0);

        float h = hsv_pixel[0] * 2.0f;
        float s = hsv_pixel[1] / 255.0f;
        float v = hsv_pixel[2] / 255.0f;

        return cv::Vec3f(h, s, v);
    }

    bool is_strong_red(float h, float s, float v) {
        bool is_hue_red = (h >= 340.0f || h <= 20.0f);
        bool is_saturated = (s >= 0.3f);
        bool is_bright_enough = (v >= 0.1f);
        return is_hue_red && is_saturated && is_bright_enough;
    }

    bool is_strong_green(float h, float s, float v) {
        bool is_hue_green = (h >= 105.0f && h <= 135.0f);
        bool is_saturated = (s >= 0.3f);
        bool is_bright_enough = (v >= 0.1f);
        return is_hue_green && is_saturated && is_bright_enough;
    }

    bool is_strong_blue(float h, float s, float v) {
        bool is_hue_blue = (h >= 200.0f && h <= 240.0f);
        bool is_saturated = (s >= 0.4f);
        bool is_bright_enough = (v >= 0.2f);
        return is_hue_blue && is_saturated && is_bright_enough;
    }

    bool is_strong_yellow(float h, float s, float v) {
        bool is_hue_yellow = (h >= 35.0f && h <= 65.0f);
        bool is_saturated = (s >= 0.5f);
        bool is_bright_enough = (v >= 0.3f);
        return is_hue_yellow && is_saturated && is_bright_enough;
    }

    cv::Mat get_mask(const cv::Mat& image, std::function<bool(float, float, float)> color_function) {
        CV_Assert(image.type() == CV_8UC3);

        int height = image.rows;
        int width = image.cols;
        cv::Mat mask(height, width, CV_8U, cv::Scalar(0));

        for (int y = 0; y < height; ++y) {
            const cv::Vec3b* row_ptr = image.ptr<cv::Vec3b>(y);
            uchar* mask_ptr = mask.ptr<uchar>(y);

            for (int x = 0; x < width; ++x) {
                cv::Vec3f hsv = bgr_to_hsv(row_ptr[x]);
                float h = hsv[0];
                float s = hsv[1];
                float v = hsv[2];

                if (color_function(h, s, v)) {
                    mask_ptr[x] = 255;
                }
            }
        }

        return mask;
    }
}
