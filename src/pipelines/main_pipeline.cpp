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
    std::vector<std::string> folders = {
        "../traffic_sign_images/vf",
        "../traffic_sign_images/vfa",
        "../traffic_sign_images/vfs",
         "../traffic_sign_images/stop",
        //"../traffic_sign_images/debug",
         //"../traffic_sign_images/problemkinder"
    };

    std::unordered_map<std::string, std::vector<cv::Mat>> images = pipeline_preprocessing::start_preprocessing_pipeline(folders);
    std::vector<cv::Mat> resized_images = images["resized"];
    std::vector<cv::Mat> color_images = images["color"];
    std::vector<cv::Mat> shape_images = images["shape"];
    std::vector<cv::Mat> stop_templates = images["stop_templates"];
    std::vector<cv::Mat> vf_templates = images["vf_templates"];
    std::vector<cv::Mat> vfa_templates = images["vfa_templates"];
    std::vector<cv::Mat> vfs_templates = images["vfs_templates"];
    std::unordered_map<std::string, std::vector<cv::Mat>> templates = {
      //  {"stop", stop_templates},
          {"vf", vf_templates},
      //  {"vfa", vfa_templates},
      //  {"vfs", vfs_templates}
    };


    /*cv::Mat img1, img2, img3, img4;
    cv::Mat arrow = basic_ops::load_image("../traffic_sign_templates/vf_signs/arrow.jpg");
    double height = arrow.rows;
    double width = arrow.cols;
    cv::resize(arrow, img1, cv::Size(width * 0.3, height * 0.3), cv::INTER_AREA);
    cv::resize(arrow, img2, cv::Size(width * 0.5, height * 0.5), cv::INTER_AREA);
    cv::resize(arrow, img3, cv::Size(width * 0.7, height * 0.7), cv::INTER_AREA);
    cv::resize(arrow, img4, cv::Size(width * 0.9, height * 0.9), cv::INTER_AREA);
    basic_ops::save_image(img1, "../traffic_sign_templates/vf_signs/arrows/arrow_30x37.jpg");
    basic_ops::save_image(img2, "../traffic_sign_templates/vf_signs/arrows/arrow_50x63.jpg");
    basic_ops::save_image(img3, "../traffic_sign_templates/vf_signs/arrows/arrow_70x88.jpg");
    basic_ops::save_image(img4, "../traffic_sign_templates/vf_signs/arrows/arrow_90x113.jpg");*/

    bool debug_mode = false;
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
