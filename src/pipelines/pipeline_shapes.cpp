#include <opencv2/opencv.hpp>
#include <vector>
#include "../header/bounding_box.hpp"
#include "../header/shape_detection.hpp"
#include "../header/basic_image_operations.hpp"

namespace shape_pipeline {
    std::vector<BoundingBox> start_pipeline_shapes(std::vector<cv::Mat> shape_images) {

        std::vector<BoundingBox> shape_bounding_boxes;

        for (size_t i = 0; i < shape_images.size(); i++) {
            const cv::Mat& image = shape_images[i];
            std::vector<std::vector<cv::Point>> contours;
            findContours(image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
            int height = image.rows;
            int width = image.cols;
            cv::Vec3b box_color = {255, 255, 255};

            int min_box_area = 500;//static_cast<int>(pow(height * 0.0275, 2));
            int max_box_area = height * width;

            std::vector<std::vector<cv::Point>> filtered_contours;
            for (const auto& contour : contours) {
                double area = contourArea(contour);
                if (area >= min_box_area && area <= max_box_area) {
                    filtered_contours.push_back(contour);
                }
            }

            std::vector<BoundingBox> bounding_boxes = bounding_box::create_bounding_boxes(
                filtered_contours, i, min_box_area, max_box_area, box_color, ""
            );

            shape_bounding_boxes.insert(
                shape_bounding_boxes.end(),
                bounding_boxes.begin(),
                bounding_boxes.end()
            );

            //debug
            /*cv::Mat image_copy = image.clone();
            cv::cvtColor(image_copy, image_copy, cv::COLOR_GRAY2BGR);
            for (const BoundingBox& bounding_box : bounding_boxes) {
                bounding_box::draw_bounding_box(bounding_box, image_copy, {0, 255, 0});
            }
            basic_ops::show_image(image_copy, "Image " + std::to_string(i));*/
        }

        shape_bounding_boxes = bounding_box::merge_duplicate_boxes(shape_bounding_boxes, 10);

        std::cout << "Shape Bounding Boxes: " << shape_bounding_boxes.size() << std::endl;
        for (auto& bbox : shape_bounding_boxes) {
            std::cout << bbox.to_string() << std::endl;
        }
        std::cout << "\n" << std::endl;

        return shape_bounding_boxes;
    }
}