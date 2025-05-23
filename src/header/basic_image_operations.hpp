#ifndef BASIC_IMAGE_OPERATIONS_HPP
#define BASIC_IMAGE_OPERATIONS_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace basic_ops {
    cv::Mat create_image(int width, int height, int channels, int gray_value);

    cv::Mat create_image_with_gradient(int width, int height, int brightness);

    cv::Mat load_image(const std::string& image_path, bool print=true);

    std::vector<cv::Mat> load_images(const std::string& folder_path, int amount, bool print=true);

    void save_image(const cv::Mat& image, const std::string& save_path, bool print=true);

    void delete_image(const std::string& image_path, bool print=true);

    void show_image(const cv::Mat& image, const std::string& title, bool print=true);

    void create_ppm_image(int width, int height, const std::string& name, const std::string& file_format);

    cv::Mat load_ppm_image(const std::string& image_path);
}
    #endif

