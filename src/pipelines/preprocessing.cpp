#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>
#include <filesystem>  // C++17
#include "../header/basic_image_operations.hpp"
#include "../header/geometrical_image_operations.hpp"
#include "../pipelines/pipeline_colors.cpp"

namespace fs = std::filesystem;

using namespace std;

int main() {

    vector<string> folders = {
        "../traffic_sign_images/vf",
        "../traffic_sign_images/vfa",
        "../traffic_sign_images/vfs",
        "../traffic_sign_images/stop"
    };

    vector<cv::Mat> original_images;
    for (const auto& folder : folders) {
        vector<cv::Mat> images = basic_ops::load_images(folder, 100);
        for (auto& image : images) {
            original_images.push_back(image);
        }
    }

    vector<cv::Mat> resized_images;
    for (const auto& image : original_images) {
        int height = image.size().height;
        int width = image.size().width;
        cv::Mat resized_image = geo_ops::resize_image(image, static_cast<int>(width/8), static_cast<int>(height/8));

        resized_images.push_back(resized_image);
    }

    vector<cv::Mat> color_images;
    for (auto& image : resized_images) {
        cv::medianBlur(image, image, 3);
        color_images.push_back(image);
    }

    vector<cv::Mat> shape_images;
    for (auto& image : color_images) {
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        cv::blur(image, image, cv::Size(5, 5));

        cv::Mat sobelX, sobelY, sobelMag;
        cv::Sobel(image, sobelX, CV_64F, 1, 0, 3);
        cv::Sobel(image, sobelY, CV_64F, 0, 1, 3);
        cv::magnitude(sobelX, sobelY, image);
        image.convertTo(image, CV_8U);

        cv::threshold(image, image, 30, 255, cv::THRESH_BINARY);
        shape_images.push_back(image);
    }

    for (auto& image : shape_images) {
        //basic_ops::show_image(image, "shape_image");
    }

    vector<cv::Mat> stop_templates = basic_ops::load_images("../traffic_sign_templates/stop_signs/resized", 100);
    vector<cv::Mat> vf_templates = basic_ops::load_images("../traffic_sign_templates/vf_signs/resized", 100);
    vector<cv::Mat> vfa_templates = basic_ops::load_images("../traffic_sign_templates/vfa_signs/resized", 100);
    vector<cv::Mat> vfs_templates = basic_ops::load_images("../traffic_sign_templates/vfs_signs/resized", 100);

    int i = color_pipeline::start_pipeline_colors(color_images, resized_images);

    return 0;
}
