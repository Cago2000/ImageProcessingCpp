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
        std::vector<BoundingBox> fused_bounding_boxes = bounding_box::fuse_bounding_box_matches(
            color_bounding_boxes, shape_bounding_boxes, 20
        );
        std::vector<BoundingBox> filtered_bounding_boxes = bounding_box::merge_duplicate_boxes(fused_bounding_boxes, 20);

        std::vector<BoundingBox> template_bounding_boxes = bounding_boxes["template"];
        template_bounding_boxes = bounding_box::merge_duplicate_boxes(template_bounding_boxes, 20);

        filtered_bounding_boxes = bounding_box::tag_bounding_boxes(fused_bounding_boxes, template_bounding_boxes, 5);
        filtered_bounding_boxes = bounding_box::merge_duplicate_boxes(filtered_bounding_boxes, 30);

        for (auto& color_bbox: color_bounding_boxes) {
            if (color_bbox.box_color == cv::Vec3b(0, 255, 255)) {
                color_bbox.box_sign = "vfs";
                filtered_bounding_boxes.push_back(color_bbox);
            }
        }
        filtered_bounding_boxes = bounding_box::merge_duplicate_boxes(filtered_bounding_boxes, 30);
        filtered_bounding_boxes = bounding_box::sort_bbox_list(filtered_bounding_boxes);

        std::vector<cv::Mat> bbox_images = std::move(resized_images);
        for (auto& bounding_box : filtered_bounding_boxes) {
            float aspectRatio = static_cast<float>(bounding_box.box_width) / static_cast<float>(bounding_box.box_height);
            if (aspectRatio < 0.6 || aspectRatio > 1.4){continue;}
            cv::Mat bbox_image = bounding_box::draw_bounding_box(bounding_box, bbox_images[bounding_box.image_index], 3, bounding_box.box_color);
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
