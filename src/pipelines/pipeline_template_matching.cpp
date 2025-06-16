#include "../header/pipeline_template_matching.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include "../header/bounding_box.hpp"
#include "../header/basic_image_operations.hpp"
#include "../header/geometrical_image_operations.hpp"

namespace template_pipeline {
    std::vector<BoundingBox> start_pipeline_template_matching(std::vector<cv::Mat> shape_images, std::vector<std::vector<cv::Mat>> templates) {
        std::vector<BoundingBox> template_matching_bounding_boxes;
        for (size_t i = 0; i < shape_images.size(); i++) {
            std::vector<std::vector<cv::Point>> contours;
            const cv::Mat& image = shape_images[i];
            int height = image.rows;
            int width = image.cols;
            cv::Vec3b box_color = {255, 255, 255};

            int min_box_area = static_cast<int>(pow(height * 0.055, 2));
            int max_box_area = height * width;

            std::vector<int> rotation_angles = {0};

            cv::Mat gray_image;
            cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

            for (const auto& template_group : templates) {
                for (const auto& template_img : template_group) {
                    for (int angle : rotation_angles) {
                        cv::Mat rotated_template = geo_ops::rotate_image(template_img, angle);
                        cv::Mat mask;
                        cv::threshold(rotated_template, mask, 250, 255, cv::THRESH_BINARY_INV);

                        int result_cols = gray_image.cols - rotated_template.cols + 1;
                        int result_rows = gray_image.rows - rotated_template.rows + 1;
                        cv::Mat result(result_rows, result_cols, CV_32FC1);

                        cv::matchTemplate(gray_image, rotated_template, result, cv::TM_CCOEFF_NORMED, mask);
                        //cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

                        double minVal, maxVal;
                        cv::Point minLoc, maxLoc, matchLoc;
                        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
                        matchLoc = maxLoc;

                        std::cout << "Confidence (maxVal): " << maxVal
                              << " | Angle: " << angle
                              << " | MatchLoc: " << matchLoc
                              << " | Template Size: " << rotated_template.cols << "x" << rotated_template.rows
                              << std::endl;

                        if(maxVal < 0.15) {
                            continue;
                        }

                        cv::Mat display;
                        cv::cvtColor(gray_image, display, cv::COLOR_GRAY2BGR);
                        cv::rectangle(display, matchLoc, cv::Point(matchLoc.x + template_img.cols, matchLoc.y + template_img.rows), cv::Scalar(0, 255, 0), 2);

                        cv::imshow("Match Result", result);
                        cv::imshow("Detected Match", display);
                        cv::imshow("Template", rotated_template);
                        cv::waitKey(0);



                        int half_width = rotated_template.cols / 2;
                        int half_height = rotated_template.rows / 2;

                        std::vector<cv::Point> box_contour = {
                            cv::Point(matchLoc.x - half_width, matchLoc.y - half_height),
                            cv::Point(matchLoc.x + half_width, matchLoc.y - half_height),
                            cv::Point(matchLoc.x + half_width, matchLoc.y + half_height),
                            cv::Point(matchLoc.x - half_width, matchLoc.y + half_height)
                        };
                        contours.push_back(box_contour);
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