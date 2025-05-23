#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "../header/colors.hpp"
#include "../header/color_detection.hpp"
#include "../header/basic_image_operations.hpp"
#include "../header/bounding_box.hpp"

namespace color_pipeline {
    std::vector<BoundingBox> start_pipeline_colors(std::vector<cv::Mat> color_images) {
        std::vector<bool(*)(float, float, float)> color_functions = {
            colors::is_strong_red,
            colors::is_strong_yellow,
            colors::is_strong_blue
        };
        std::vector<BoundingBox> color_bounding_boxes;

        for (size_t i = 0; i < color_images.size(); i++) {
            const cv::Mat& image = color_images[i];
            int height = image.size().height;
            int width = image.size().width;
            int min_box_area = static_cast<int>((height * 0.055) * (height * 0.055));
            int max_box_area = height * width;
            for (auto color_function : color_functions) {
                cv::Mat mask = colors::get_mask(image, color_function);
                const std::vector<std::vector<cv::Point>>& blobs = cd::get_blobs(mask);
                cv::Vec3b box_color = colors::get_color_from_function(color_function);
                std::vector<BoundingBox> bounding_boxes = bounding_box::create_bounding_boxes(blobs, i, min_box_area, max_box_area, box_color);
                for(auto bounding_box: bounding_boxes) {
                    color_bounding_boxes.push_back(bounding_box);
                }
            }
        }
        color_bounding_boxes = bounding_box::merge_duplicate_boxes(color_bounding_boxes, 10);

        std::cout << "Color Bounding Boxes: " << color_bounding_boxes.size() << std::endl;
        for (auto& bbox:color_bounding_boxes) {
            std::cout << bbox.to_string() << std::endl;
        }
        std::cout << "\n" << std::endl;
        return color_bounding_boxes;
    }
}
