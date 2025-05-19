#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include "../header/bounding_box.hpp"
#include "../header/basic_image_operations.hpp"  // assuming similar utility
#include "../header/pipeline_box_fusion.hpp"

namespace box_fusion_pipeline {
    std::vector<BoundingBox> start_pipeline_box_fusion(std::vector<BoundingBox> color_bounding_boxes, std::vector<BoundingBox> shape_bounding_boxes, std::vector<cv::Mat> resized_images) {
         std::vector<BoundingBox> bounding_boxes = bounding_box::fuse_bounding_box_matches(
            color_bounding_boxes, shape_bounding_boxes, 15
        );

        bounding_boxes = bounding_box::merge_duplicate_boxes(bounding_boxes, 20);

        for (const auto& bounding_box : bounding_boxes) {
            //std::cout << bounding_box.to_string() << std::endl;
        }

        std::vector<cv::Mat> bbox_images = resized_images;
        for (auto& bounding_box : bounding_boxes) {
            cv::Mat bbox_image = bounding_box::draw_bounding_box(bounding_box, bbox_images[bounding_box.image_index]);
            bbox_images[bounding_box.image_index] = bbox_image;
        }

        for (const auto& img : bbox_images) {
            basic_ops::show_image(img, "traffic_sign_bboxes", false);
        }
        return bounding_boxes;
    }
}
