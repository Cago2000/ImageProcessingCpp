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

            std::vector<std::vector<cv::Point>> contours = sd::get_contours(image, 15);

            int height = image.rows;
            int width = image.cols;
            cv::Vec3b box_color = {255, 255, 255};

            int min_box_area = static_cast<int>(pow(height * 0.055, 2));
            int max_box_area = height * width;

            std::vector<BoundingBox> bounding_boxes = bounding_box::create_bounding_boxes(contours, i, min_box_area, max_box_area, box_color);
            shape_bounding_boxes.insert(shape_bounding_boxes.end(), bounding_boxes.begin(), bounding_boxes.end());
        }

        shape_bounding_boxes = bounding_box::merge_duplicate_boxes(shape_bounding_boxes, 10);

        return shape_bounding_boxes;
    }
}