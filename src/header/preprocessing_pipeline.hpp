#ifndef PREPROCESSING_PIPELINE_HPP
#define PREPROCESSING_PIPELINE_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace pipeline_preprocessing {

    std::vector<cv::Mat> preprocess_resizing(const std::vector<cv::Mat>& images);
    std::vector<cv::Mat> preprocess_colors(const std::vector<cv::Mat>& images);
    std::vector<cv::Mat> preprocess_shapes(const std::vector<cv::Mat>& images);
    std::vector<std::vector<cv::Mat>> start_preprocessing_pipeline();

}

#endif // PREPROCESSING_PIPELINE_HPP
