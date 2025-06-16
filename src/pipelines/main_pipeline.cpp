#include "../header/main_pipeline.hpp"
#include "../header/bounding_box.hpp"
#include "../header/pipeline_colors.hpp"
#include "../header/pipeline_shapes.hpp"
#include "../header/pipeline_template_matching.hpp"
#include "../header/pipeline_box_fusion.hpp"

#include <opencv2/opencv.hpp>

#include "../header/preprocessing_pipeline.hpp"

int main() {

    std::unordered_map<std::string, std::vector<cv::Mat>> images = pipeline_preprocessing::start_preprocessing_pipeline();
    std::vector<cv::Mat> resized_images = images["resized"];
    std::vector<cv::Mat> color_images = images["color"];
    std::vector<cv::Mat> shape_images = images["shape"];
    std::vector<cv::Mat> stop_templates = images["stop_templates"];
    std::vector<cv::Mat> vf_templates = images["vf_templates"];
    std::vector<cv::Mat> vfa_templates = images["vfa_templates"];
    std::vector<cv::Mat> vfs_templates = images["vfs_templates"];
    std::vector<std::vector<cv::Mat>> templates = {stop_templates, vf_templates, vfa_templates, vfs_templates};

    std::vector<BoundingBox> color_bounding_boxes = color_pipeline::start_pipeline_colors(color_images);
    std::vector<BoundingBox> shape_bounding_boxes = shape_pipeline::start_pipeline_shapes(shape_images);
    std::vector<BoundingBox> template_matching_bounding_boxes =  template_pipeline::start_pipeline_template_matching(color_images, templates);
    std::unordered_map<std::string, std::vector<BoundingBox>> bounding_boxes ={
        {"color", color_bounding_boxes},
        {"shape", shape_bounding_boxes},
        {"template", template_matching_bounding_boxes}
    };
    std::vector<BoundingBox> fused_bounding_boxes = box_fusion_pipeline::start_pipeline_box_fusion(bounding_boxes, resized_images);
}
