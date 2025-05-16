#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <numeric> // for std::accumulate

struct BoundingBox {
    int center_y;
    int center_x;
    std::vector<int> box_corners; // top, left, bottom, right
    int box_height;
    int box_width;
    int box_area;
    cv::Vec3b box_color; // BGR color
    int image_index;

    BoundingBox(int y, int x, const std::vector<int> corners, int height, int width, int area, const cv::Vec3b& color, int idx)
        : center_y(y), center_x(x), box_corners(corners), box_height(height), box_width(width), box_area(area), box_color(color), image_index(idx) {}

};

namespace bounding_box {

    BoundingBox* create_bounding_box(const std::vector<cv::Point>& blob, int image_index, int min_box_area, int max_box_area, cv::Vec3b box_color) {
        if (blob.empty()) return nullptr;

        int left = std::numeric_limits<int>::max();
        int right = std::numeric_limits<int>::min();
        int top = std::numeric_limits<int>::max();
        int bottom = std::numeric_limits<int>::min();

        for (const auto& pt : blob) {
            left = std::min(left, pt.x);
            right = std::max(right, pt.x);
            top = std::min(top, pt.y);
            bottom = std::max(bottom, pt.y);
        }


        int width = right - left + 1;
        int height = bottom - top + 1;
        int area = width * height;

        std::cout << top << " " << left << " " << bottom << " " << right << " " << height << " " << width <<std::endl;

        if (area < min_box_area || area > max_box_area) return nullptr;

        double aspect_ratio = std::max(static_cast<double>(width)/height, static_cast<double>(height)/width);
        if (aspect_ratio > 1.75) return nullptr;

        int center_y = (top + bottom) / 2;
        int center_x = (left + right) / 2;

        std::vector<int> box_corners = {top, left, bottom, right};

        return new BoundingBox(center_y, center_x, box_corners, height, width, area, box_color, image_index);
    }

    std::vector<BoundingBox> create_bounding_boxes(const std::vector<std::vector<cv::Point>>& blobs,
                                               int image_index, int min_box_area, int max_box_area,
                                               cv::Vec3b& box_color) {
        std::vector<BoundingBox> bounding_boxes;
        for (const auto& blob : blobs) {
            BoundingBox* bbox = create_bounding_box(blob, image_index, min_box_area, max_box_area, box_color);
            if (bbox != nullptr) {
                bounding_boxes.push_back(*bbox);
                delete bbox;
            }
        }
        return bounding_boxes;
    }

    // Draw bounding box on image
    cv::Mat draw_bounding_box(const BoundingBox& box, cv::Mat& image) {
        int top = box.box_corners[0];
        int left = box.box_corners[1];
        int bottom = box.box_corners[2];
        int right = box.box_corners[3];

        std::cout << top << " " << left << " " << bottom << " " << right << std::endl;

        // Draw horizontal lines
        for (int x = left; x <= right; ++x) {
            image.at<cv::Vec3b>(top, x) = box.box_color;
            image.at<cv::Vec3b>(bottom, x) = box.box_color;
        }
        // Draw vertical lines
        for (int y = top; y <= bottom; ++y) {
            image.at<cv::Vec3b>(y, left) = box.box_color;
            image.at<cv::Vec3b>(y, right) = box.box_color;
        }
        return image;
    }

    // Fuse bounding boxes between two lists
    std::vector<BoundingBox> fuse_bounding_box_matches(const std::vector<BoundingBox>& boxes1, const std::vector<BoundingBox>& boxes2, int max_deviation) {
        std::vector<BoundingBox> new_boxes;

        for (const auto& box1 : boxes1) {
            for (const auto& box2 : boxes2) {
                if (std::abs(box1.center_y - box2.center_y) >= max_deviation || std::abs(box1.center_x - box2.center_x) >= max_deviation) {
                    continue; // too far apart
                }

                std::vector<int> new_corners(4);
                for (int i = 0; i < 4; ++i) {
                    new_corners[i] = (box1.box_corners[i] + box2.box_corners[i]) / 2;
                }

                int new_center_y = (box1.center_y + box2.center_y) / 2;
                int new_center_x = (box1.center_x + box2.center_x) / 2;
                int new_height = (box1.box_height + box2.box_height) / 2;
                int new_width = (box1.box_width + box2.box_width) / 2;
                int new_area = new_height * new_width;

                cv::Vec3b new_color = {255, 255, 255};
                if (box1.box_color != cv::Vec3b(255, 255, 255)) new_color = box1.box_color;
                if (box2.box_color != cv::Vec3b(255, 255, 255)) new_color = box2.box_color;

                int new_image_index = box1.image_index;

                new_boxes.emplace_back(new_center_y, new_center_x, new_corners, new_height, new_width, new_area, new_color, new_image_index);
            }
        }

        return new_boxes;
    }

    std::vector<BoundingBox> merge_duplicate_boxes(const std::vector<BoundingBox>& boxes, int max_deviation) {
        std::vector<BoundingBox> merged_boxes;
        std::vector<bool> visited(boxes.size(), false);

        for (size_t i = 0; i < boxes.size(); ++i) {
            if (visited[i]) continue;

            std::vector<const BoundingBox*> similar_boxes = { &boxes[i] };
            visited[i] = true;

            for (size_t j = i + 1; j < boxes.size(); ++j) {
                if (visited[j]) continue;
                if (std::abs(boxes[i].center_y - boxes[j].center_y) <= max_deviation &&
                    std::abs(boxes[i].center_x - boxes[j].center_x) <= max_deviation) {
                    similar_boxes.push_back(&boxes[j]);
                    visited[j] = true;
                    }
            }
            std::vector<int> avg_corners(4, 0);
            for (int c = 0; c < 4; ++c) {
                int sum_c = 0;
                for (auto b : similar_boxes) {
                    sum_c += b->box_corners[c];
                }
                avg_corners[c] = sum_c / (int)similar_boxes.size();
            }

            int avg_center_y = std::accumulate(similar_boxes.begin(), similar_boxes.end(), 0, [](int sum, const BoundingBox* b){ return sum + b->center_y; }) / (int)similar_boxes.size();
            int avg_center_x = std::accumulate(similar_boxes.begin(), similar_boxes.end(), 0, [](int sum, const BoundingBox* b){ return sum + b->center_x; }) / (int)similar_boxes.size();
            int avg_height = std::accumulate(similar_boxes.begin(), similar_boxes.end(), 0, [](int sum, const BoundingBox* b){ return sum + b->box_height; }) / (int)similar_boxes.size();
            int avg_width = std::accumulate(similar_boxes.begin(), similar_boxes.end(), 0, [](int sum, const BoundingBox* b){ return sum + b->box_width; }) / (int)similar_boxes.size();
            int avg_area = avg_height * avg_width;
            int image_index = similar_boxes[0]->image_index;

            cv::Vec3b avg_color = {255, 255, 255};
            for (auto b : similar_boxes) {
                if (b->box_color != cv::Vec3b(255, 255, 255)) {
                    avg_color = b->box_color;
                    break;
                }
            }

            merged_boxes.emplace_back(avg_center_y, avg_center_x, avg_corners, avg_height, avg_width, avg_area, avg_color, image_index);
        }

        return merged_boxes;
    }
}
