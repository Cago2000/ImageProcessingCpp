#include <opencv2/opencv.hpp>
#include <vector>
#include "../header/bounding_box.hpp"
#include "../header/basic_image_operations.hpp"

namespace shape_pipeline {
    std::vector<BoundingBox> start_pipeline_shapes(std::vector<cv::Mat> shape_images, bool debug_mode) {
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
                    std::string shape;
                    cv::approxPolyDP(contour, approx, epsilon * cv::arcLength(contour, true), true);
                    cv::Rect rect = cv::boundingRect(approx);
                    float aspectRatio = (float)rect.width / rect.height;
                    if (aspectRatio < 0.25 || aspectRatio > 4.0){continue;}
                    const size_t vertices = approx.size();
                    switch (vertices) {
                        case 3:
                            shape = "Triangle";
                            break;
                        case 4: {
                                shape = "Square or Diamond";
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
            if (!debug_mode) {continue;}
            cv::Mat image_copy = image.clone();
            cv::Mat image_copy_with_boxes = image.clone();
            cv::cvtColor(image_copy_with_boxes, image_copy_with_boxes, cv::COLOR_GRAY2BGR);
            for (const BoundingBox& bounding_box : shape_bounding_boxes) {
                if (bounding_box.image_index == i) {
                    std::cout << bounding_box.to_string() << std::endl;
                    if (bounding_box.box_shape == "Octagon") {
                        bounding_box::draw_bounding_box(bounding_box, image_copy_with_boxes, {255, 0, 0});
                    }
                    else if (bounding_box.box_shape == "Triangle") {
                        bounding_box::draw_bounding_box(bounding_box, image_copy_with_boxes, {0, 0, 255});
                    }
                    else if (bounding_box.box_shape == "Rectangle" || bounding_box.box_shape == "Square or Diamond") {
                        bounding_box::draw_bounding_box(bounding_box, image_copy_with_boxes, {0, 255, 255});
                    }
                }
            }
            std::cout << std::endl;
            cv::imshow("Image " + std::to_string(i), image_copy);
            basic_ops::show_image(image_copy_with_boxes, "Image " + std::to_string(i)+ " with Boxes");
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