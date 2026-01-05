#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>
#include <algorithm>
#include <numeric>
#include "header/bounding_box.hpp"
#include "header/basic_image_operations.hpp"
#include <memory_resource>

namespace bounding_box {
    BoundingBox *create_bounding_box(const std::vector<cv::Point> &contour, int image_index, const cv::Vec3b &box_color, const std::string &shape, std::string sign) {
        if (contour.empty()) return nullptr;

        int left = std::numeric_limits<int>::max();
        int right = std::numeric_limits<int>::min();
        int top = std::numeric_limits<int>::max();
        int bottom = std::numeric_limits<int>::min();

        for (const auto& pt : contour) {
            left = std::min(left, pt.x);
            right = std::max(right, pt.x);
            top = std::min(top, pt.y);
            bottom = std::max(bottom, pt.y);
        }

        int width = right - left + 1;
        int height = bottom - top + 1;
        int area = width * height;

        int center_y = (top + bottom) / 2;
        int center_x = (left + right) / 2;

        std::vector<int> box_corners = {top, left, bottom, right};

        auto* bbox = new BoundingBox(center_y, center_x, box_corners, height, width, area, box_color, shape, std::move(sign), image_index);
        return bbox;
    }

    std::vector<BoundingBox> create_bounding_boxes(const std::vector<std::vector<cv::Point>>& contours,
                                               int image_index, std::string shape,
                                               cv::Vec3b& box_color, std::string sign) {
        std::vector<BoundingBox> bounding_boxes;
        for (const auto& contour : contours) {
            BoundingBox* bbox = create_bounding_box(contour, image_index, box_color,  shape,sign);
            if (bbox != nullptr) {
                bounding_boxes.push_back(*bbox);
                delete bbox;
            }
        }
        return bounding_boxes;
    }

    cv::Mat draw_bounding_box(const BoundingBox& box, cv::Mat& image, const cv::Vec3b& color = {255, 255, 255}) {

        int top = box.box_corners[0];
        int left = box.box_corners[1];
        int bottom = box.box_corners[2];
        int right = box.box_corners[3];


        for (int x = left; x <= right; ++x) {
            image.at<cv::Vec3b>(top, x) = color;
            image.at<cv::Vec3b>(bottom, x) = color;
        }
        for (int y = top; y <= bottom; ++y) {
            image.at<cv::Vec3b>(y, left) = color;
            image.at<cv::Vec3b>(y, right) = color;
        }
        return image;
    }

std::map<std::string, int> shape_complexity = {
    {"Triangle", 3},
    {"Square or Diamond", 4},
    {"Rectangle", 4},
    {"Pentagon", 5},
    {"Hexagon", 6},
    {"Heptagon", 7},
    {"Octagon", 8},
    {"Circle", 100}
};

std::vector<BoundingBox> fuse_bounding_box_matches(
    const std::vector<BoundingBox>& boxes1,
    const std::vector<BoundingBox>& boxes2,
    int max_deviation)
{
    std::vector<BoundingBox> new_boxes;

    for (const auto& box1 : boxes1) {
        for (const auto& box2 : boxes2) {
            if (box1.image_index != box2.image_index){
                continue;
            }
            if (std::abs(box1.center_y - box2.center_y) >= max_deviation ||
                std::abs(box1.center_x - box2.center_x) >= max_deviation) {
                continue;
            }

            cv::Rect rect1(box1.center_x - box1.box_width / 2, box1.center_y - box1.box_height / 2,
                           box1.box_width, box1.box_height);
            cv::Rect rect2(box2.center_x - box2.box_width / 2, box2.center_y - box2.box_height / 2,
                           box2.box_width, box2.box_height);

            bool box1_inside_box2 = (rect2.contains(rect1.tl()) && rect2.contains(rect1.br()));
            bool box2_inside_box1 = (rect1.contains(rect2.tl()) && rect1.contains(rect2.br()));

            const BoundingBox* dominant_box = nullptr;

            if (box1_inside_box2 || box2_inside_box1) {
                dominant_box = (box1.box_area >= box2.box_area) ? &box1 : &box2;
            }

            std::vector<int> new_corners(4);
            int new_center_y, new_center_x, new_height, new_width, new_area;

            if (dominant_box) {
                new_corners = dominant_box->box_corners;
                new_center_y = dominant_box->center_y;
                new_center_x = dominant_box->center_x;
                new_height = dominant_box->box_height;
                new_width = dominant_box->box_width;
                new_area = dominant_box->box_area;
            } else {
                for (int i = 0; i < 4; ++i)
                    new_corners[i] = (box1.box_corners[i] + box2.box_corners[i]) / 2;

                new_center_y = (box1.center_y + box2.center_y) / 2;
                new_center_x = (box1.center_x + box2.center_x) / 2;
                new_height = (box1.box_height + box2.box_height) / 2;
                new_width = (box1.box_width + box2.box_width) / 2;
                new_area = new_height * new_width;
            }

            cv::Vec3b new_color = {255, 255, 255};
            if (box1.box_color != cv::Vec3b(255, 255, 255)) new_color = box1.box_color;
            if (box2.box_color != cv::Vec3b(255, 255, 255)) new_color = box2.box_color;

            std::string new_shape;
            int v1 = shape_complexity.count(box1.box_shape) ? shape_complexity[box1.box_shape] : INT_MAX;
            int v2 = shape_complexity.count(box2.box_shape) ? shape_complexity[box2.box_shape] : INT_MAX;
            new_shape = (v1 <= v2) ? box1.box_shape : box2.box_shape;

            int new_image_index = box1.image_index;
            std::string new_box_sign = box1.box_sign.empty() ? box2.box_sign : box1.box_sign;

            new_boxes.emplace_back(new_center_y, new_center_x, new_corners, new_height, new_width,
                                   new_area, new_color, new_shape, new_box_sign, new_image_index);
        }
    }

    return new_boxes;
}



std::vector<BoundingBox> merge_duplicate_boxes(const std::vector<BoundingBox>& boxes, int max_deviation) {
    std::vector<BoundingBox> merged_boxes;
    std::vector<bool> visited(boxes.size(), false);

    std::map<std::string, int> shape_complexity = {
        {"Triangle", 3},
        {"Square or Diamond", 4},
        {"Rectangle", 4},
        {"Pentagon", 5},
        {"Hexagon", 6},
        {"Heptagon", 7},
        {"Octagon", 8},
        {"Circle", 100}
    };

    std::unordered_map<std::string, int> sign_complexity = {
        {"vf", 1},
        {"vfa", 2},
        {"vfs", 3},
        {"stop", 4},
        {"circle", 100}
    };

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (visited[i]) continue;

        std::vector similar_boxes = {&boxes[i]};
        visited[i] = true;

        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (visited[j]) continue;

            if (boxes[i].image_index == boxes[j].image_index &&
                std::abs(boxes[i].center_y - boxes[j].center_y) <= max_deviation &&
                std::abs(boxes[i].center_x - boxes[j].center_x) <= max_deviation) {
                similar_boxes.push_back(&boxes[j]);
                visited[j] = true;
            }
        }

        std::vector<int> avg_corners(4, 0);
        for (int c = 0; c < 4; ++c) {
            int sum = 0;
            for (const auto* b : similar_boxes) {
                sum += b->box_corners[c];
            }
            avg_corners[c] = sum / static_cast<int>(similar_boxes.size());
        }

        int avg_center_y = std::accumulate(similar_boxes.begin(), similar_boxes.end(), 0,
            [](int sum, const BoundingBox* b) { return sum + b->center_y; }) / similar_boxes.size();

        int avg_center_x = std::accumulate(similar_boxes.begin(), similar_boxes.end(), 0,
            [](int sum, const BoundingBox* b) { return sum + b->center_x; }) / similar_boxes.size();

        int avg_height = std::accumulate(similar_boxes.begin(), similar_boxes.end(), 0,
            [](int sum, const BoundingBox* b) { return sum + b->box_height; }) / similar_boxes.size();

        int avg_width = std::accumulate(similar_boxes.begin(), similar_boxes.end(), 0,
            [](int sum, const BoundingBox* b) { return sum + b->box_width; }) / similar_boxes.size();

        int avg_area = avg_height * avg_width;

        cv::Vec3b avg_color = {255, 255, 255};
        for (const auto* b : similar_boxes) {
            if (b->box_color != cv::Vec3b(255, 255, 255)) {
                avg_color = b->box_color;
                break;
            }
        }
        std::string least_complex_shape;
        int min_vertices = INT_MAX;
        for (const auto* b : similar_boxes) {
            int complexity = shape_complexity.count(b->box_shape) ? shape_complexity[b->box_shape] : INT_MAX;
            if (complexity < min_vertices) {
                min_vertices = complexity;
                least_complex_shape = b->box_shape;
            }
        }
        std::string least_complex_sign;
        int min_sign_rank = INT_MAX;
        for (const auto* b : similar_boxes) {
            if (b->box_sign.empty()) continue;
            int rank = sign_complexity.count(b->box_sign) ? sign_complexity[b->box_sign] : INT_MAX;
            if (rank < min_sign_rank) {
                min_sign_rank = rank;
                least_complex_sign = b->box_sign;
            }
        }
        int image_index = similar_boxes[0]->image_index;
        merged_boxes.emplace_back(
            avg_center_y, avg_center_x, avg_corners,
            avg_height, avg_width, avg_area,
            avg_color, least_complex_shape, least_complex_sign,
            image_index
        );
    }
    return merged_boxes;
}

std::vector<BoundingBox> tag_bounding_boxes(std::vector<BoundingBox>& fused_bounding_boxes, const std::vector<BoundingBox>& template_bounding_boxes, int clipping_tolerance) {
    std::vector<BoundingBox> filtered_bounding_boxes;

    for (auto& bounding_box : fused_bounding_boxes) {
        if (!bounding_box.box_sign.empty()) {
            continue;
        }
        if (bounding_box.box_color == cv::Vec3b(0, 0, 255) && bounding_box.box_shape == "Triangle") {
            bounding_box.box_sign = "vfa";

            for (const auto& template_box : template_bounding_boxes) {
                if (template_box.image_index != bounding_box.image_index) {continue;}
                bool template_inside =
                    template_box.box_corners[0] >= bounding_box.box_corners[0] - clipping_tolerance &&
                    template_box.box_corners[1] >= bounding_box.box_corners[1] - clipping_tolerance &&
                    template_box.box_corners[2] <= bounding_box.box_corners[2] + clipping_tolerance &&
                    template_box.box_corners[3] <= bounding_box.box_corners[3] + clipping_tolerance;

                if (template_inside) {
                    bounding_box.box_sign = "vf";
                    break;
                }
            }
        }
        if (bounding_box.box_color == cv::Vec3b(0, 255, 255)){
            bounding_box.box_sign = "vfs";
        }
        if (bounding_box.box_color == cv::Vec3b(0, 0, 255) && bounding_box.box_shape == "Octagon") {
            bounding_box.box_sign = "stop";
        }
        if (!bounding_box.box_sign.empty()) {
            filtered_bounding_boxes.push_back(bounding_box);
        }
    }

    return filtered_bounding_boxes;
}


    std::vector<cv::Mat> get_roi(const std::vector<cv::Mat>& images, const std::vector<BoundingBox>& bounding_boxes, int min_area, int margin) {
        std::vector<cv::Mat> cropped_images;
        for (const auto& bounding_box: bounding_boxes) {
            if (bounding_box.box_area < min_area) {continue;}
            cv::Mat color_image_copy = images[bounding_box.image_index].clone();
            draw_bounding_box(bounding_box, color_image_copy, bounding_box.box_color);
            cv::Point topLeft(bounding_box.box_corners[1]-margin, bounding_box.box_corners[0]-margin);
            cv::Point bottomRight(bounding_box.box_corners[3]+margin, bounding_box.box_corners[2]+margin);
            cv::Rect roi(topLeft, bottomRight);
            cv::Mat crop = images[bounding_box.image_index].clone()(roi);
            cropped_images.push_back(crop);
        }
        return cropped_images;
    }

    std::vector<BoundingBox> sort_bbox_list(std::vector<BoundingBox> bounding_boxes) {
        std::sort(bounding_boxes.begin(), bounding_boxes.end(),
        [](const BoundingBox& a, const BoundingBox& b) {
         return a.image_index < b.image_index;
        });
        return bounding_boxes;
    }

}
