#include <opencv2/opencv.hpp>
#include <vector>
#include "../header/bounding_box.hpp"
#include "../header/shape_detection.hpp"
#include "../header/basic_image_operations.hpp"

namespace shape_pipeline {
    std::vector<BoundingBox> start_pipeline_shapes(std::vector<cv::Mat> shape_images) {
        std::vector<double> epsilons = {0.03};
        std::vector<BoundingBox> shape_bounding_boxes;
        for (size_t i = 0; i < shape_images.size(); i++) {
            const cv::Mat& image = shape_images[i];
            std::vector<std::vector<cv::Point>> contours;
            findContours(image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
            int height = image.rows;
            int width = image.cols;
            cv::Vec3b box_color = {255, 255, 255};

            int min_box_area = 2000;
            int max_box_area = height * width;

            for (const auto& contour : contours) {
                const double area = cv::boundingRect(contour).area();
                if (area < min_box_area || area > max_box_area){continue;}
                for (const auto& epsilon : epsilons) {
                    std::vector<cv::Point> approx;
                    cv::approxPolyDP(contour, approx, epsilon * cv::arcLength(contour, true), true);
                    std::string shape;

                    const size_t vertices = approx.size();
                    switch (vertices) {
                        case 3:
                            shape = "Triangle";
                            break;
                        case 4: {
                            cv::Rect rect = cv::boundingRect(approx);
                            float aspectRatio = (float)rect.width / rect.height;
                            if (aspectRatio > 0.90 && aspectRatio < 1.10)
                                shape = "Square or Diamond";
                            else
                                shape = "Rectangle";
                            break;
                        }
                        case 8:
                            shape = "Octagon";
                            break;
                        default:
                            continue;
                    }
                    BoundingBox* bounding_box = bounding_box::create_bounding_box(contour, i, box_color, shape, "");
                    if (bounding_box != nullptr) {
                        shape_bounding_boxes.push_back(*bounding_box);
                        delete bounding_box;
                    }
                }
            }
            //debug
            cv::Mat image_copy = image.clone();
            std::vector<BoundingBox> bounding_boxes = shape_bounding_boxes;
            cv::cvtColor(image_copy, image_copy, cv::COLOR_GRAY2BGR);
            for (const BoundingBox& bounding_box : bounding_boxes) {
                if (bounding_box.image_index == i) {
                    bounding_box::draw_bounding_box(bounding_box, image_copy, {0, 255, 0});
                }
            }
            basic_ops::show_image(image_copy, "Image " + std::to_string(i));
        }

        shape_bounding_boxes = bounding_box::merge_duplicate_boxes(shape_bounding_boxes, 20);

        std::cout << "Shape Bounding Boxes: " << shape_bounding_boxes.size() << std::endl;
        for (auto& bbox : shape_bounding_boxes) {
            std::cout << bbox.to_string() << std::endl;
        }
        std::cout << "\n" << std::endl;

        return shape_bounding_boxes;
    }
}