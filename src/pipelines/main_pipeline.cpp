#include "../header/main_pipeline.hpp"
#include "../header/bounding_box.hpp"
#include "../header/pipeline_colors.hpp"
#include "../header/pipeline_shapes.hpp"
#include "../header/pipeline_template_matching.hpp"
#include "../header/pipeline_box_fusion.hpp"
#include <opencv2/opencv.hpp>
#include "../header/basic_image_operations.hpp"
#include "../header/preprocessing_pipeline.hpp"

int main() {
    int resize_factor;
    bool debug_mode = true;

    /* Bounding Box Colors:
     * VFS = YELLOW
     * VF = BLACK
     * VFA = BLUE
     * STOP = RED
    */

    // CLEAN IMAGES, resize_factor is 1!
    /*std::vector<std::string> folders = {
      "../traffic_sign_templates/clean_traffic_signs",
    };
    resize_factor = 1;*/


    // UNCLEAN IMAGES, change resize_factor to 8!
    std::vector<std::string> folders = {
          "../traffic_sign_images/vf",
          "../traffic_sign_images/vfa",
          "../traffic_sign_images/vfs",
           "../traffic_sign_images/stop",
      };
    resize_factor = 8;

    std::unordered_map<std::string, std::vector<cv::Mat>> images =
        pipeline_preprocessing::start_preprocessing_pipeline(folders, resize_factor);
    std::vector<cv::Mat> resized_images = images["resized"];
    std::vector<cv::Mat> color_images = images["color"];
    std::vector<cv::Mat> shape_images = images["shape"];
    std::vector<cv::Mat> stop_templates = images["stop_templates"];
    std::vector<cv::Mat> vf_templates = images["vf_templates"];
    std::vector<cv::Mat> vfa_templates = images["vfa_templates"];
    std::vector<cv::Mat> vfs_templates = images["vfs_templates"];
    std::unordered_map<std::string, std::vector<cv::Mat>> templates = {
          {"vf", vf_templates},
    };

    std::vector<BoundingBox> color_bounding_boxes = color_pipeline::start_pipeline_colors(color_images, debug_mode);
    std::vector<BoundingBox> shape_bounding_boxes = shape_pipeline::start_pipeline_shapes(shape_images, debug_mode);
    std::vector<BoundingBox> template_matching_bounding_boxes =  template_pipeline::start_pipeline_template_matching(resized_images, templates, debug_mode);
    std::unordered_map<std::string, std::vector<BoundingBox>> bounding_boxes ={
        {"color", color_bounding_boxes},
        {"shape", shape_bounding_boxes},
        {"template", template_matching_bounding_boxes}
    };
    std::vector<BoundingBox> fused_bounding_boxes = box_fusion_pipeline::start_pipeline_box_fusion(bounding_boxes, resized_images);
}
