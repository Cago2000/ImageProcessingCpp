#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "../header/colors.hpp"
#include "../header/color_detection.hpp"
#include "../header/basic_image_operations.hpp"
#include "../header/bounding_box.hpp"

namespace color_pipeline {
    std::vector<BoundingBox> start_pipeline_colors(std::vector<cv::Mat> color_images, bool debug_mode) {
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
            int min_box_area = 1000;
            int max_box_area = height * width;
            for (auto color_function : color_functions) {
                cv::Mat mask = colors::get_mask(image, color_function);
                const std::vector<std::vector<cv::Point>>& blobs = cd::get_blobs(mask);
                cv::Vec3b box_color = colors::get_color_from_function(color_function);
                for (const auto& blob : blobs) {
                    int left = std::numeric_limits<int>::max();
                    int right = std::numeric_limits<int>::min();
                    int top = std::numeric_limits<int>::max();
                    int bottom = std::numeric_limits<int>::min();
                    for (const auto& pt : blob) {
                        left = std::min(left, pt.x);
                        right = std::max(right, pt.x);
                        top = std::min(top, pt.y);
                        bottom = std::max(bottom, pt.y);
                    }
                    const int blob_width = right - left + 1;
                    const int blob_height = bottom - top + 1;
                    int area = blob_width * blob_height;
                    if (area < min_box_area || area > max_box_area){continue;}
                    BoundingBox* bounding_box = bounding_box::create_bounding_box(blob, i, box_color, "", "");
                    if (bounding_box != nullptr) {
                        color_bounding_boxes.push_back(*bounding_box);
                    }
                }
                if (debug_mode) {
                    cv::imshow("Mask #" + std::to_string(i), mask);
                    cv::waitKey(0);
                    cv::destroyAllWindows();
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
