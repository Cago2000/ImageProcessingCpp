#ifndef STATISTICAL_OPERATIONS_HPP
#define STATISTICAL_OPERATIONS_HPP

#include <vector>
#include <functional>
#include <cstdint>
#include <opencv2/opencv.hpp> // For cv::Mat

namespace stat_ops {

    // Applies a simple Gaussian-like filter (box blur) to a grayscale image.
    // Returns an empty cv::Mat if input is invalid (color image or even kernel size).
    cv::Mat gauss_filter(const cv::Mat& image, int dim);

    // Counts co-occurrences in an image using a relation function.
    // relation_function receives (const cv::Mat&, int x, int y) and returns bool.
    int co_occurrence(const cv::Mat& image, std::function<bool(const cv::Mat&, int, int)> relation_function);

    // Calculates the median pixel intensity.
    uint8_t median(const cv::Mat& image);

    // Calculates the mean pixel intensity.
    double mean(const cv::Mat& image);

    // Calculates the variance of pixel intensities.
    double variance(const cv::Mat& image);

    // Calculates the standard deviation of pixel intensities.
    double stddev(const cv::Mat& image);

    // Computes histogram for grayscale image.
    // Returns empty vector if input image is not single channel.
    std::vector<uint32_t> histogram(const cv::Mat& image);

    // Computes relative histogram (normalized histogram).
    std::vector<double> relative_histogram(const cv::Mat& image);

    // Computes cumulative histogram from relative histogram.
    std::vector<double> cumulative_histogram(const cv::Mat& image);

    // Calculates entropy of the image.
    double entropy(const cv::Mat& image);

    // Performs histogram equalization on a grayscale image.
    // Returns empty cv::Mat if input is not grayscale.
    cv::Mat histogram_equalization(const cv::Mat& image);

    // Performs gamma correction on a grayscale image.
    cv::Mat gamma_equalization(const cv::Mat& image, double gamma);

} // namespace stat_ops

#endif // STATISTICAL_OPERATIONS_HPP
