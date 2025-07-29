#ifndef SHAPES_HPP
#define SHAPES_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace shapes {
    bool is_valid_octagon(const std::vector<cv::Point>& approx);
}



#endif //SHAPES_HPP
