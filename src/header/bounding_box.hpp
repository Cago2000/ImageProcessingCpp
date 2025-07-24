#ifndef BOUNDING_BOX_HPP
#define BOUNDING_BOX_HPP
#include <utility>
#include <vector>
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
    std::string box_sign;
    int image_index;

    BoundingBox(int y, int x, const std::vector<int> corners, int height, int width, int area,
                const cv::Vec3b& color, std::string shape, std::string sign, int index)
        : center_y(y), center_x(x), box_corners(corners), box_height(height), box_width(width),
          box_area(area), box_color(color), box_shape(std::move(shape)), box_sign(std::move(sign)), image_index(index) {}

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
            << ", sign=" << box_sign
            << ")";
        return oss.str();
    }
};

namespace bounding_box {
    BoundingBox *create_bounding_box(const std::vector<cv::Point> &contour, int image_index, const cv::Vec3b &box_color,
                                     double max_aspect_ratio, const std::string &shape, std::string sign);

    std::vector<BoundingBox> create_bounding_boxes(const std::vector<std::vector<cv::Point>>& contours, int image_index,
                                                    cv::Vec3b& box_color, std::string sign);

    cv::Mat draw_bounding_box(const BoundingBox& bounding_box, cv::Mat& image, const cv::Vec3b& color);

    std::vector<BoundingBox> fuse_bounding_box_matches(const std::vector<BoundingBox>& boxes1,
                                                       const std::vector<BoundingBox>& boxes2, int max_deviation);

    std::vector<BoundingBox> merge_duplicate_boxes(const std::vector<BoundingBox>& boxes, int max_deviation);

    std::vector<cv::Mat> get_roi(const std::vector<cv::Mat>& images, const std::vector<BoundingBox>& bounding_boxes, int min_area, int margin);
}

#endif // BOUNDING_BOX_HPP

