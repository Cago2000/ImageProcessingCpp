#ifndef SHAPE_PIPELINE_HPP
#define SHAPE_PIPELINE_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include "../header/bounding_box.hpp"


namespace shape_pipeline {
    std::vector<BoundingBox> start_pipeline_shapes(std::vector<cv::Mat> shape_images, std::vector<cv::Mat> resized_images);
}

#endif // SHAPE_PIPELINE_HPP
