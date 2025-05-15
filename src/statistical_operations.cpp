#include "header/statistical_operations.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>

cv::Mat gauss_filter(const cv::Mat& image, int dim) {
    if (image.channels() != 1 || dim % 2 == 0) return cv::Mat();

    int half_dim = dim / 2;
    cv::Mat result = image.clone();
    int height = image.rows;
    int width = image.cols;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::vector<uint16_t> pixels;
            for (int a = -half_dim; a <= half_dim; ++a) {
                for (int b = -half_dim; b <= half_dim; ++b) {
                    int ny = y + a;
                    int nx = x + b;
                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        pixels.push_back(image.at<uint8_t>(ny, nx));
                    }
                }
            }
            uint32_t sum = std::accumulate(pixels.begin(), pixels.end(), 0u);
            result.at<uint8_t>(y, x) = static_cast<uint8_t>(sum / pixels.size());
        }
    }
    return result;
}

int co_occurrence(const cv::Mat& image, std::function<bool(const cv::Mat&, int, int)> relation_function) {
    int height = image.rows;
    int width = image.cols;
    int count = 0;

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            if (relation_function(image, x, y))
                ++count;

    return count;
}

uint8_t median(const cv::Mat& image) {
    std::vector<uint8_t> pixels;
    pixels.reserve(image.rows * image.cols);

    for (int y = 0; y < image.rows; ++y)
        for (int x = 0; x < image.cols; ++x)
            pixels.push_back(image.at<uint8_t>(y, x));

    std::nth_element(pixels.begin(), pixels.begin() + pixels.size()/2, pixels.end());
    return pixels[pixels.size()/2];
}

double mean(const cv::Mat& image) {
    double sum = 0;
    int total = image.rows * image.cols;
    for (int y = 0; y < image.rows; ++y)
        for (int x = 0; x < image.cols; ++x)
            sum += image.at<uint8_t>(y, x);

    return sum / total;
}

double variance(const cv::Mat& image) {
    double m = mean(image);
    double sum_sq = 0;
    int total = image.rows * image.cols;
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            double diff = image.at<uint8_t>(y, x) - m;
            sum_sq += diff * diff;
        }
    }
    return sum_sq / total;
}

double stddev(const cv::Mat& image) {
    return std::sqrt(variance(image));
}

std::vector<uint32_t> histogram(const cv::Mat& image) {
    if (image.channels() != 1) return {};

    int max_val = 255; // For uint8_t
    std::vector<uint32_t> hist(max_val + 1, 0);

    for (int y = 0; y < image.rows; ++y)
        for (int x = 0; x < image.cols; ++x)
            hist[image.at<uint8_t>(y, x)]++;

    return hist;
}

// Compute relative histogram (normalized) for a single-channel image
std::vector<double> relative_histogram(const cv::Mat& image) {
    CV_Assert(image.channels() == 1);
    int hist_size = 256;  // assuming 8-bit image, adjust if needed

    std::vector<int> hist(hist_size, 0);
    int total_pixels = image.rows * image.cols;

    for (int r = 0; r < image.rows; ++r) {
        const uchar* ptr = image.ptr<uchar>(r);
        for (int c = 0; c < image.cols; ++c) {
            ++hist[ptr[c]];
        }
    }

    std::vector<double> relative_hist(hist_size);
    for (int i = 0; i < hist_size; ++i) {
        relative_hist[i] = static_cast<double>(hist[i]) / total_pixels;
    }
    return relative_hist;
}

// Compute cumulative histogram
std::vector<double> cumulative_histogram(const cv::Mat& image) {
    CV_Assert(image.channels() == 1);
    auto rel_hist = relative_histogram(image);

    std::vector<double> cum_hist(rel_hist.size(), 0.0);
    double cumulative_sum = 0.0;
    for (size_t i = 0; i < rel_hist.size(); ++i) {
        cumulative_sum += rel_hist[i];
        cum_hist[i] = cumulative_sum;
    }
    return cum_hist;
}

// Histogram equalization
cv::Mat histogram_equalization(const cv::Mat& image) {
    CV_Assert(image.channels() == 1);

    auto cum_hist = cumulative_histogram(image);

    // Build lookup table (mapping old gray values to new ones)
    std::vector<uchar> lookup_table(cum_hist.size());
    for (size_t i = 0; i < cum_hist.size(); ++i) {
        lookup_table[i] = static_cast<uchar>(std::floor(cum_hist[i] * 255.0));
    }

    cv::Mat equalized_image = image.clone();

    for (int r = 0; r < image.rows; ++r) {
        const uchar* src_ptr = image.ptr<uchar>(r);
        uchar* dst_ptr = equalized_image.ptr<uchar>(r);
        for (int c = 0; c < image.cols; ++c) {
            dst_ptr[c] = lookup_table[src_ptr[c]];
        }
    }
    return equalized_image;
}

// Gamma equalization (gamma correction)
cv::Mat gamma_equalization(const cv::Mat& image, double gamma) {
    CV_Assert(image.channels() == 1);

    int max_value = 255; // assuming 8-bit image
    std::vector<uchar> lookup_table(max_value + 1);

    for (int i = 0; i <= max_value; ++i) {
        double normalized = static_cast<double>(i) / max_value;
        lookup_table[i] = static_cast<uchar>(std::floor(std::pow(normalized, gamma) * max_value));
    }

    cv::Mat gamma_image = image.clone();

    for (int r = 0; r < image.rows; ++r) {
        const uchar* src_ptr = image.ptr<uchar>(r);
        uchar* dst_ptr = gamma_image.ptr<uchar>(r);
        for (int c = 0; c < image.cols; ++c) {
            dst_ptr[c] = lookup_table[src_ptr[c]];
        }
    }
    return gamma_image;
}

