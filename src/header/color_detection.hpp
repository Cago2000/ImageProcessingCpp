#ifndef COLOR_DETECTION_HPP
#define COLOR_DETECTION_HPP

#include <vector>
#include <utility>
#include <opencv2/opencv.hpp>

using Coord = std::pair<int, int>;
using Blob = std::vector<Coord>;
namespace cd {
    std::vector<std::vector<cv::Point>> get_blobs(cv::Mat mask);

#endif // COLOR_DETECTION_HPP
}
