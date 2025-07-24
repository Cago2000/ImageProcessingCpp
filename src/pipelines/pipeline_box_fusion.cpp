#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>
#include <iostream>
#include "../header/bounding_box.hpp"
#include "../header/basic_image_operations.hpp"  // assuming similar utility
#include "../header/pipeline_box_fusion.hpp"

namespace box_fusion_pipeline {
    std::vector<BoundingBox> start_pipeline_box_fusion(std::unordered_map<std::string, std::vector<BoundingBox>> bounding_boxes, std::vector<cv::Mat> resized_images) {

        std::vector<BoundingBox> color_bounding_boxes = bounding_boxes["color"];
        std::vector<BoundingBox> shape_bounding_boxes = bounding_boxes["shape"];
        std::vector<BoundingBox> template_bounding_boxes = bounding_boxes["template"];

        std::vector<BoundingBox> fused_bounding_boxes = bounding_box::fuse_bounding_box_matches(
            color_bounding_boxes, shape_bounding_boxes, 20
        );

        std::vector<BoundingBox> filtered_bounding_boxes;
        for (auto& bounding_box : fused_bounding_boxes) {
            if (!bounding_box.box_sign.empty()) {
                continue;
            }

            if (bounding_box.box_color == cv::Vec3b(0, 0, 255) && bounding_box.box_shape == "Triangle") {
                bounding_box.box_sign = "vfa";
                for (auto const& template_bounding_box: template_bounding_boxes) {
                    if (abs(template_bounding_box.center_x-bounding_box.center_x) < 50 &&
                        abs(template_bounding_box.center_y-bounding_box.center_y) < 50) {
                        bounding_box.box_sign = "vf";
                    }
                }
            }
            if (bounding_box.box_color == cv::Vec3b(0, 255, 255) &&
                (bounding_box.box_shape == "Square or Diamond" || bounding_box.box_shape == "Rectangle")){
                bounding_box.box_sign = "vfs";
            }
            if (bounding_box.box_color == cv::Vec3b(0, 0, 255) && bounding_box.box_shape == "Octagon") {
                bounding_box.box_sign = "stop";
            }
            if (!bounding_box.box_sign.empty()) {
                filtered_bounding_boxes.push_back(bounding_box);
            }
        }

        filtered_bounding_boxes = bounding_box::merge_duplicate_boxes(filtered_bounding_boxes, 20);

        std::vector<cv::Mat> bbox_images = std::move(resized_images);
        for (auto& bounding_box : filtered_bounding_boxes) {
            cv::Mat bbox_image = bounding_box::draw_bounding_box(bounding_box, bbox_images[bounding_box.image_index], bounding_box.box_color);
            bbox_images[bounding_box.image_index] = bbox_image;
        }

        std::cout << "Bounding Boxes: " << filtered_bounding_boxes.size() << std::endl;
        for (auto& bounding_box: filtered_bounding_boxes) {
            std::cout << bounding_box.to_string() << std::endl;
        }
        std::cout << std::endl << std::endl;

        int i = 0;
        for (const auto& img : bbox_images) {
            basic_ops::show_image(img, "image " + std::to_string(i++), false);
        }
        return filtered_bounding_boxes;
    }
}
