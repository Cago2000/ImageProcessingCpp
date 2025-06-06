#include "../header/pipeline_template_matching.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include "../header/bounding_box.hpp"
#include "../header/basic_image_operations.hpp"

namespace template_pipeline {
    std::vector<BoundingBox> start_pipeline_template_matching(std::vector<cv::Mat> shape_images, std::vector<std::vector<cv::Mat>> templates) {
        std::vector<BoundingBox> template_matching_bounding_boxes;
        float confidence_threshold = 0.5f;
        for (size_t i = 0; i < shape_images.size(); i++) {
            std::vector<std::vector<cv::Point>> contours;
            const cv::Mat& image = shape_images[i];
            int height = image.rows;
            int width = image.cols;
            cv::Vec3b box_color = {255, 255, 255};

            int min_box_area = static_cast<int>(pow(height * 0.055, 2));
            int max_box_area = height * width;

            std::vector<double> rotation_angles = {-5, -3, 0, 3, 5};

            for (const auto& template_group : templates) {
                for (const auto& template_img : template_group) {
                    for (double angle : rotation_angles) {
                        cv::Mat rotated_template;
                        cv::Point2f center(template_img.cols / 2.0F, template_img.rows / 2.0F);
                        cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
                        cv::warpAffine(template_img, rotated_template, rot_mat, template_img.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

                        if (rotated_template.rows > image.rows || rotated_template.cols > image.cols)
                            continue;

                        cv::Mat match_result;
                        cv::matchTemplate(image, rotated_template, match_result, cv::TM_CCOEFF_NORMED);

                        cv::Mat threshold_mask;
                        cv::threshold(match_result, threshold_mask, confidence_threshold, 255.0, cv::THRESH_BINARY);
                        threshold_mask.convertTo(threshold_mask, CV_8U);

                        std::vector<std::vector<cv::Point>> template_contours;
                        cv::findContours(threshold_mask, template_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                        for (const auto& contour : template_contours) {
                            cv::Rect bbox_rect = cv::boundingRect(contour);

                            std::vector<cv::Point> box_contour = {
                                cv::Point(bbox_rect.x, bbox_rect.y),
                                cv::Point(bbox_rect.x + bbox_rect.width, bbox_rect.y),
                                cv::Point(bbox_rect.x + bbox_rect.width, bbox_rect.y + bbox_rect.height),
                                cv::Point(bbox_rect.x, bbox_rect.y + bbox_rect.height)
                            };

                            contours.push_back(box_contour);
                        }
                    }
                }
            }
                std::vector<BoundingBox> bounding_boxes = bounding_box::create_bounding_boxes(contours, i, min_box_area, max_box_area, box_color, "Unknown");
                template_matching_bounding_boxes.insert(template_matching_bounding_boxes.end(), bounding_boxes.begin(), bounding_boxes.end());
        }

        template_matching_bounding_boxes = bounding_box::merge_duplicate_boxes(template_matching_bounding_boxes, 10);

        std::cout << "Template Matching Bounding Boxes: " << template_matching_bounding_boxes.size() << std::endl;
        for (auto& bbox : template_matching_bounding_boxes) {
            std::cout << bbox.to_string() << std::endl;
        }
        std::cout << "\n" << std::endl;

        return template_matching_bounding_boxes;
    }
}