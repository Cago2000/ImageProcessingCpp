#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>
#include <filesystem>  // C++17
#include <bits/regex_constants.h>

#include "../header/basic_image_operations.hpp"
#include "../header/geometrical_image_operations.hpp"
#include "../header/pipeline_box_fusion.hpp"
#include "../header/pipeline_colors.hpp"
#include "../header/pipeline_shapes.hpp"

namespace fs = std::filesystem;

std::vector<cv::Mat> preprocess_resizing(const std::vector<cv::Mat>& images) {
    std::vector<cv::Mat> resized_images;
    for (const auto& image : images) {
        int height = image.size().height;
        int width = image.size().width;
        cv::Mat resized_image = geo_ops::resize_image(image, static_cast<int>(width/8), static_cast<int>(height/8));

        resized_images.push_back(resized_image);
    }
    return resized_images;
}

std::vector<cv::Mat> preprocess_colors(const std::vector<cv::Mat>& images) {
    std::vector<cv::Mat> color_images;
    for (const auto& image : images) {
        cv::Mat color_image;
        cv::medianBlur(image, color_image, 5);
        color_images.push_back(color_image);
    }
    return color_images;
}

std::vector<cv::Mat> preprocess_shapes(const std::vector<cv::Mat>& images) {
    std::vector<cv::Mat> shape_images;
    for (auto& image : images) {
        cv::Mat shape_image;
        cv::cvtColor(image, shape_image, cv::COLOR_BGR2GRAY);
        cv::blur(shape_image, shape_image, cv::Size(5, 5));

        cv::Mat sobelX, sobelY, sobelMag;
        cv::Sobel(shape_image, sobelX, CV_64F, 1, 0, 3);
        cv::Sobel(shape_image, sobelY, CV_64F, 0, 1, 3);
        cv::magnitude(sobelX, sobelY, shape_image);
        shape_image.convertTo(shape_image, CV_8U);

        cv::threshold(shape_image, shape_image, 30, 255, cv::THRESH_BINARY);
        shape_images.push_back(shape_image);
    }
    return shape_images;
}

int main() {

    std::vector<std::string> folders = {
        "../traffic_sign_images/vf",
        "../traffic_sign_images/vfa",
        "../traffic_sign_images/vfs",
        "../traffic_sign_images/stop"

    };

    std::vector<cv::Mat> original_images;
    for (const auto& folder : folders) {
        std::vector<cv::Mat> images = basic_ops::load_images(folder, 100, true);
        for (auto& image : images) {
            original_images.push_back(image);
        }
    }

    std::vector<cv::Mat> resized_images = preprocess_resizing(original_images);
    std::vector<cv::Mat> color_images = preprocess_colors(resized_images);
    std::vector<cv::Mat> shape_images = preprocess_shapes(resized_images);


    /*std::vector<cv::Mat> stop_templates = basic_ops::load_images("../traffic_sign_templates/stop_signs/resized", 100, false);
    std::vector<cv::Mat> vf_templates = basic_ops::load_images("../traffic_sign_templates/vf_signs/resized", 100, false);
    std::vector<cv::Mat> vfa_templates = basic_ops::load_images("../traffic_sign_templates/vfa_signs/resized", 100, false);
    std::vector<cv::Mat> vfs_templates = basic_ops::load_images("../traffic_sign_templates/vfs_signs/resized", 100, false);

    cv::Mat stop = basic_ops::load_image("../traffic_sign_templates/stop_signs/stop.jpg", false);
    cv::Mat vf = basic_ops::load_image("../traffic_sign_templates/vf_signs/vf.jpg", false);
    cv::Mat vfa = basic_ops::load_image("../traffic_sign_templates/vfa_signs/vfa.jpg", false);
    cv::Mat vfs = basic_ops::load_image("../traffic_sign_templates/vfs_signs/vfs.jpg", false);
    std::vector<cv::Mat> base_templates = {stop, vf, vfa, vfs};

    std::vector<cv::Mat> resized_template_images = base_templates;
    std::vector<cv::Mat> colors_template_images = preprocess_colors(base_templates);
    std::vector<cv::Mat> shape_template_images = preprocess_shapes(base_templates);*/

    std::vector<BoundingBox> color_bounding_boxes = color_pipeline::start_pipeline_colors(color_images);
    std::cout << "Color Bounding Boxes: " << color_bounding_boxes.size() << std::endl;
    for (auto& bbox:color_bounding_boxes) {
        std::cout << bbox.to_string() << std::endl;
    }

    std::vector<BoundingBox> shape_bounding_boxes = shape_pipeline::start_pipeline_shapes(shape_images);
    std::cout << "Shape Bounding Boxes: " << shape_bounding_boxes.size() << std::endl;
    for (auto& bbox:shape_bounding_boxes) {
        std::cout << bbox.to_string() << std::endl;
    }
    std::cout << "\n" << std::endl;

    std::vector<BoundingBox> bounding_boxes = box_fusion_pipeline::start_pipeline_box_fusion(color_bounding_boxes, shape_bounding_boxes, resized_images);
    std::cout << "Bounding Boxes: " <<bounding_boxes.size() << std::endl;
    for (auto& bbox:bounding_boxes) {
        std::cout << bbox.to_string() << std::endl;
    }
    std::cout << "\n" << std::endl;
    return 0;
}
