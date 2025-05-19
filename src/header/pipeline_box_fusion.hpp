#ifndef PIPELINE_BOX_FUSION_H
#define PIPELINE_BOX_FUSION_H

#include "../header/bounding_box.hpp"

namespace box_fusion_pipeline {
    std::vector<BoundingBox> start_pipeline_box_fusion(std::vector<BoundingBox> color_bounding_boxes, std::vector<BoundingBox> shape_bounding_boxes, std::vector<cv::Mat> resized_images);
}

#endif //PIPELINE_BOX_FUSION_H
