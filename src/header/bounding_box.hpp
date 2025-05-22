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
    std::vector<int> box_corners;
    int box_height;
    int box_width;
    int box_area;
    cv::Vec3b box_color;
    std::string box_shape;
    int image_index;

    BoundingBox(int y, int x, std::vector<int> corners, int height, int width, int area,
                 cv::Vec3b box_color, std::string shape, int image_index);

    std::string to_string() const {
        std::ostringstream oss;
        oss << "BoundingBox(image_index=" << image_index
            << ", center=(" << center_y << ", " << center_x << ")"
            << ", corners=[top:" << box_corners[0]
            << ", left:" << box_corners[1]
            << ", bottom:" << box_corners[2]
            << ", right:" << box_corners[3] << "]"
            << ", height=" << box_height
            << ", width=" << box_width
            << ", area=" << box_area
            << ", color=(B:" << static_cast<int>(box_color[0])
            << ", G:" << static_cast<int>(box_color[1])
            << ", R:" << static_cast<int>(box_color[2]) << ")"
            << ", shape=" << box_shape
            << ")";
        return oss.str();
    }
};

namespace bounding_box {
    BoundingBox create_bounding_box(const std::vector<cv::Point>& blob, int image_index,
                                                   int min_box_area, int max_box_area, cv::Vec3b& box_color);

    std::vector<BoundingBox> create_bounding_boxes(const std::vector<std::vector<cv::Point>>& blobs, int image_index,
                                                   int min_box_area, int max_box_area, cv::Vec3b& box_color);

    cv::Mat draw_bounding_box(const BoundingBox& bounding_box, cv::Mat& image);

    std::vector<BoundingBox> fuse_bounding_box_matches(const std::vector<BoundingBox>& boxes1,
                                                       const std::vector<BoundingBox>& boxes2, int max_deviation);

    std::vector<BoundingBox> merge_duplicate_boxes(const std::vector<BoundingBox>& boxes, int max_deviation);
}

#endif // BOUNDING_BOX_HPP

