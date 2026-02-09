#include "../header/pipeline_template_matching.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include "../header/bounding_box.hpp"
#include "../header/basic_image_operations.hpp"
#include "../header/geometrical_image_operations.hpp"

namespace template_pipeline {
    std::vector<BoundingBox> start_pipeline_template_matching(std::vector<cv::Mat> images, const std::unordered_map<std::string, std::vector<cv::Mat>>& templates, bool debug_mode) {
        std::vector<BoundingBox> template_matching_bounding_boxes;
        std::vector<BoundingBox> bounding_boxes;
        for (size_t i = 0; i < images.size(); i++) {
            std::vector<std::vector<cv::Point>> contours;
            const cv::Mat& image = images[i];
            int height = image.rows;
            int width = image.cols;
            cv::Vec3b box_color = {255, 255, 255};

            int min_box_area = static_cast<int>(pow(height * 0.055, 2));
            int max_box_area = height * width;

            std::vector<int> rotation_angles = {-5, -3, 0, 3, 5};
            std::vector<double> horizontal_stretches = {1.0 ,0.8, 0.6};

            cv::Mat gray_image;
            cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
            for (const auto& [sign, template_group] : templates) {
                if (debug_mode) {std::cout << sign << std::endl;}
                for (const auto& template_img : template_group){
                    int template_width = template_img.cols;
                    int template_height = template_img.rows;
                    for (int angle : rotation_angles) {
                        for (double stretch : horizontal_stretches) {
                            cv::Mat stretched_template;
                            cv::resize(template_img, stretched_template, cv::Size(static_cast<int>(template_width*stretch), template_height), cv::INTER_AREA);
                            cv::Mat rotated_template = geo_ops::rotate_image_cv(stretched_template, angle);
                            cv::Mat mask;
                            cv::threshold(rotated_template, mask, 230, 255, cv::THRESH_BINARY_INV);

                            std::vector<std::vector<cv::Point>> point;
                            cv::findContours(mask, point, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                            mask = cv::Mat::zeros(mask.size(), CV_8UC1);
                            cv::fillPoly(mask, point, cv::Scalar(255));

                            cv::Mat mask_copy = mask.clone();

                            int result_cols = gray_image.cols - rotated_template.cols + 1;
                            int result_rows = gray_image.rows - rotated_template.rows + 1;
                            cv::Mat result(result_rows, result_cols, CV_32FC1);

                            cv::matchTemplate(gray_image, rotated_template, result, cv::TM_CCOEFF_NORMED, mask);
                            //cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

                            double minVal, maxVal;
                            cv::Point minLoc, maxLoc, matchLoc;
                            cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
                            matchLoc = maxLoc;

                            cv::Mat display;
                            cv::cvtColor(gray_image, display, cv::COLOR_GRAY2BGR);
                            cv::rectangle(display, matchLoc, cv::Point(matchLoc.x + template_img.cols, matchLoc.y + template_img.rows), cv::Scalar(0, 255, 0), 2);



                            if(maxVal < 0.70 || maxVal > 1.0) {
                                continue;
                            }

                            if (debug_mode) {
                                std::cout << "Confidence (maxVal): " << maxVal
                                          << " | Angle: " << angle
                                          << " | MatchLoc: " << matchLoc
                                          << " | Template Size: "
                                          << rotated_template.cols << "x" << rotated_template.rows
                                          << std::endl;

                                int start_x = 50;
                                int start_y = 50;
                                int padding = 100;

                                // Top-left
                                cv::namedWindow("Match Result #" + std::to_string(i), cv::WINDOW_NORMAL);
                                cv::imshow("Match Result #" + std::to_string(i), result);
                                cv::moveWindow("Match Result #" + std::to_string(i), start_x, start_y);

                                // Top-right
                                cv::namedWindow("Detected Match #" + std::to_string(i), cv::WINDOW_NORMAL);
                                cv::imshow("Detected Match #" + std::to_string(i), display);
                                cv::moveWindow(
                                    "Detected Match #" + std::to_string(i),
                                    start_x + result.cols + padding,
                                    start_y
                                );

                                // Bottom-left
                                cv::namedWindow("Template #" + std::to_string(i), cv::WINDOW_NORMAL);
                                cv::imshow("Template #" + std::to_string(i), rotated_template);
                                cv::moveWindow(
                                    "Template #" + std::to_string(i),
                                    start_x,
                                    start_y + result.rows + 2*padding
                                );

                                // Bottom-right
                                cv::namedWindow("Mask #" + std::to_string(i), cv::WINDOW_NORMAL);
                                cv::imshow("Mask #" + std::to_string(i), mask_copy);
                                cv::moveWindow(
                                    "Mask #" + std::to_string(i),
                                    start_x + result.cols + 2*padding,
                                    start_y + result.rows + 2*padding
                                );

                                cv::waitKey(1);  // force repaint
                                cv::waitKey(0);  // block
                                cv::destroyAllWindows();
                            }


                            int half_width = static_cast<int>(rotated_template.cols / 2);
                            int half_height = static_cast<int>( rotated_template.rows / 2);

                            std::vector<cv::Point> box_contour = {
                                cv::Point(matchLoc.x - half_width, matchLoc.y - half_height),
                                cv::Point(matchLoc.x + half_width, matchLoc.y - half_height),
                                cv::Point(matchLoc.x + half_width, matchLoc.y + half_height),
                                cv::Point(matchLoc.x - half_width, matchLoc.y + half_height)
                            };

                            double area = cv::contourArea(box_contour);
                            if (area < min_box_area || area > max_box_area){continue;}
                            BoundingBox* bbox = bounding_box::create_bounding_box(box_contour, i, box_color, "", sign);
                            if (bbox != nullptr) {
                                bounding_boxes.push_back(*bbox);
                                delete bbox;
                            }
                        }
                    }
                }
                if (debug_mode) {std::cout << std::endl;}
            }
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