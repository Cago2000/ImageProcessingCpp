#include "../header/main_pipeline.hpp"
#include "../header/bounding_box.hpp"
#include "../header/pipeline_colors.hpp"
#include "../header/pipeline_shapes.hpp"
#include "../header/pipeline_box_fusion.hpp"

#include <opencv2/opencv.hpp>

#include "../header/preprocessing_pipeline.hpp"

int main() {

    std::vector<std::vector<cv::Mat>> images = pipeline_preprocessing::start_preprocessing_pipeline();
    std::vector<cv::Mat> resized_images = images[0];
    std::vector<cv::Mat> color_images = images[1];
    std::vector<cv::Mat> shape_images = images[2];

    std::vector<BoundingBox> color_bounding_boxes = color_pipeline::start_pipeline_colors(color_images);
    std::cout << "Color Bounding Boxes: " << color_bounding_boxes.size() << std::endl;
    for (auto& bbox:color_bounding_boxes) {
        std::cout << bbox.to_string() << std::endl;
    }

    std::vector<BoundingBox> shape_bounding_boxes = shape_pipeline::start_pipeline_shapes(shape_images);
    std::cout << "Shape Bounding Boxes: " << shape_bounding_boxes.size() << std::endl;
    for (auto& bbox:shape_bounding_boxes) {
        std::cout << bbox.to_string() << std::endl;
    }
    std::cout << "\n" << std::endl;

    std::vector<BoundingBox> bounding_boxes = box_fusion_pipeline::start_pipeline_box_fusion(color_bounding_boxes, shape_bounding_boxes, resized_images);
    /*std::cout << "Bounding Boxes: " <<bounding_boxes.size() << std::endl;
    for (auto& bbox:bounding_boxes) {
        std::cout << bbox.to_string() << std::endl;
    }
    std::cout << "\n" << std::endl;*/
}
