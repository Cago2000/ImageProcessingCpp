#pragma once
#include <opencv2/opencv.hpp>

namespace filters {
    cv::Mat grayScaleFilter(const cv::Mat& image);
    cv::Mat blackWhiteFilter(const cv::Mat& image, int threshold);
    cv::Mat blurFilter(const cv::Mat& image, int kernelDim, int kernelIntensity);
    cv::Mat sobelFilter(const cv::Mat& image, const std::string& mode, int intensity);
    cv::Mat laplaceFilter(const cv::Mat& image, int intensity, int threshold);
    cv::Mat linearGrayScaling(cv::Mat image, float c1, float c2);
    cv::Mat isodensityFilter(const cv::Mat& image, int degree);
    cv::Mat erosion(const cv::Mat& image, int dim);
    cv::Mat dilation(const cv::Mat& image, int dim);
    cv::Mat medianFilter(const cv::Mat& image, int dim);
    cv::Mat sobelFilterFFT(const cv::Mat& image, const std::string& mode, int intensity);
    cv::Mat medianFilterSorted(const cv::Mat& image, int dim);
}
