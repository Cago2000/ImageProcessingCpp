#ifndef BOUNDING_BOX_HPP
#define BOUNDING_BOX_HPP

#include <vector>
#include <array>
#include <optional>
#include <opencv2/opencv.hpp>

class BoundingBox {
public:
    int center_y;
    int center_x;
    std::array<int, 4> box_corners;
    int box_height;
    int box_width;
    int box_area;
    std::array<int, 3> box_color;
    int image_index;

    BoundingBox(int y, int x, const std::array<int, 4>& corners, int height, int width, int area,
                cv::Vec3b box_color, int image_index);

    std::string repr() const;
};

namespace bounding_box {

    std::vector<BoundingBox> create_bounding_boxes(const std::vector<std::vector<cv::Point>>& blobs, int image_index,
                                                   int min_box_area, int max_box_area, cv::Vec3b box_color);

    BoundingBox create_bounding_box(const std::vector<cv::Point>& blob, int image_index,
                                                   int min_box_area, int max_box_area, cv::Vec3b& box_color);

    cv::Mat draw_bounding_box(const BoundingBox& bounding_box, cv::Mat image);

    std::vector<BoundingBox> fuse_bounding_box_matches(const std::vector<BoundingBox>& boxes1,
                                                       const std::vector<BoundingBox>& boxes2, int max_deviation);

    std::vector<BoundingBox> merge_duplicate_boxes(const std::vector<BoundingBox>& boxes, int max_deviation);
}

#endif // BOUNDING_BOX_HPP

