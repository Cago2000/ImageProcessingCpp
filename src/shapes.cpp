#include "header/shapes.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>

namespace shapes {
    bool is_valid_octagon(const std::vector<cv::Point>& approx) {
        if (approx.size() != 8) return false;

        std::vector<double> side_lengths;
        for (size_t i = 0; i < 8; ++i) {
            double length = cv::norm(approx[i] - approx[(i + 1) % 8]);
            side_lengths.push_back(length);
        }

        double avg_length = std::accumulate(side_lengths.begin(), side_lengths.end(), 0.0) / 8.0;
        for (double length : side_lengths) {
            if (std::abs(length - avg_length) > avg_length * 0.2)
                return false;
        }
        return true;
    }
}