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

    std::unordered_map<std::string, std::vector<cv::Mat>> images = pipeline_preprocessing::start_preprocessing_pipeline();
    std::vector<cv::Mat> resized_images = images["resized"];
    std::vector<cv::Mat> color_images = images["color"];
    std::vector<cv::Mat> shape_images = images["shape"];
    std::vector<cv::Mat> stop_templates = images["stop_templates"];
    std::vector<cv::Mat> vf_templates = images["vf_templates"];
    std::vector<cv::Mat> vfa_templates = images["vfa_templates"];
    std::vector<cv::Mat> vfs_templates = images["vfs_templates"];
    std::unordered_map<std::string, std::vector<cv::Mat>> templates = {
        {"stop", stop_templates},
        {"vf", vf_templates},
        {"vfa", vfa_templates},
        {"vfs", vfs_templates}
    };

    /*
    cv::Mat img1, img2, img3, img4;
    cv::Mat stop = basic_ops::load_image("../traffic_sign_templates/stop_signs/stop.jpg");
    cv::resize(stop, img1, cv::Size(150, 150), cv::INTER_AREA);
    cv::resize(stop, img2, cv::Size(120, 120), cv::INTER_AREA);
    cv::resize(stop, img3, cv::Size(90, 90), cv::INTER_AREA);
    cv::resize(stop, img4, cv::Size(60, 60), cv::INTER_AREA);
    basic_ops::save_image(img1, "../traffic_sign_templates/stop_signs/resized/stop_150x150.jpg");
    basic_ops::save_image(img2, "../traffic_sign_templates/stop_signs/resized/stop_120x120.jpg");
    basic_ops::save_image(img3, "../traffic_sign_templates/stop_signs/resized/stop_90x90.jpg");
    basic_ops::save_image(img4, "../traffic_sign_templates/stop_signs/resized/stop_60x60.jpg");

    cv::Mat vfa = basic_ops::load_image("../traffic_sign_templates/vfa_signs/vfa.jpg");
    cv::resize(vfa, img1, cv::Size(150, 150), cv::INTER_AREA);
    cv::resize(vfa, img2, cv::Size(120, 120), cv::INTER_AREA);
    cv::resize(vfa, img3, cv::Size(90, 90), cv::INTER_AREA);
    cv::resize(vfa, img4, cv::Size(60, 60), cv::INTER_AREA);
    basic_ops::save_image(img1, "../traffic_sign_templates/vfa_signs/resized/vfa_150x150.jpg");
    basic_ops::save_image(img2, "../traffic_sign_templates/vfa_signs/resized/vfa_120x120.jpg");
    basic_ops::save_image(img3, "../traffic_sign_templates/vfa_signs/resized/vfa_90x90.jpg");
    basic_ops::save_image(img4, "../traffic_sign_templates/vfa_signs/resized/vfa_60x60.jpg");

    cv::Mat vfs = basic_ops::load_image("../traffic_sign_templates/vfs_signs/vfs.jpg");
    cv::resize(vfs, img1, cv::Size(150, 150), cv::INTER_AREA);
    cv::resize(vfs, img2, cv::Size(120, 120), cv::INTER_AREA);
    cv::resize(vfs, img3, cv::Size(90, 90), cv::INTER_AREA);
    cv::resize(vfs, img4, cv::Size(60, 60), cv::INTER_AREA);
    basic_ops::save_image(img1, "../traffic_sign_templates/vfs_signs/resized/vfs_150x150.jpg");
    basic_ops::save_image(img2, "../traffic_sign_templates/vfs_signs/resized/vfs_120x120.jpg");
    basic_ops::save_image(img3, "../traffic_sign_templates/vfs_signs/resized/vfs_90x90.jpg");
    basic_ops::save_image(img4, "../traffic_sign_templates/vfs_signs/resized/vfs_60x60.jpg");
    */

    std::vector<BoundingBox> color_bounding_boxes = color_pipeline::start_pipeline_colors(color_images);
    std::vector<BoundingBox> shape_bounding_boxes = shape_pipeline::start_pipeline_shapes(shape_images);
    std::vector<BoundingBox> template_matching_bounding_boxes =  template_pipeline::start_pipeline_template_matching(resized_images, templates);
    std::unordered_map<std::string, std::vector<BoundingBox>> bounding_boxes ={
        {"color", color_bounding_boxes},
        {"shape", shape_bounding_boxes},
        {"template", template_matching_bounding_boxes}
    };
    std::vector<BoundingBox> fused_bounding_boxes = box_fusion_pipeline::start_pipeline_box_fusion(bounding_boxes, resized_images);
}
