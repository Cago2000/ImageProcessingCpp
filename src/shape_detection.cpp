#include "header/shape_detection.hpp"
#include <cmath>
#include <stdexcept>

namespace sd {
    bool is_edge(const cv::Mat& binary_img, int y, int x) {
        int height = binary_img.rows;
        int width = binary_img.cols;

        if (binary_img.at<uchar>(y, x) != 255)
            return false;

        const int dy[4] = {-1, 1, 0, 0};
        const int dx[4] = {0, 0, -1, 1};
        for (int i = 0; i < 4; ++i) {
            int ny = y + dy[i];
            int nx = x + dx[i];
            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                if (binary_img.at<uchar>(ny, nx) == 0) {
                    return true;
                }
            }
        }
        return false;
    }

    std::vector<cv::Point> trace_contour(const cv::Mat& binary_image, cv::Mat& visited, int y, int x) {
        const int height = binary_image.rows;
        const int width = binary_image.cols;

        std::vector<cv::Point> directions = {
            {-1, 0}, {-1, 1}, {0, 1}, {1, 1},
            {1, 0},  {1, -1}, {0, -1}, {-1, -1}
        };

        std::vector<cv::Point> contour;
        std::vector<cv::Point> stack = {{x, y}};
        visited.at<uchar>(y, x) = 1;

        while (!stack.empty()) {
            cv::Point p = stack.back();
            stack.pop_back();
            contour.push_back(p);

            for (const auto& d : directions) {
                int nx = p.x + d.x;
                int ny = p.y + d.y;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    if (visited.at<uchar>(ny, nx) == 0 && is_edge(binary_image, ny, nx)) {
                        visited.at<uchar>(ny, nx) = 1;
                        stack.push_back({nx, ny});
                        break;
                    }
                }
            }
        }
        return contour;
    }

    std::vector<std::vector<cv::Point>> get_contours(const cv::Mat& binary_image, int angle_tolerance) {
        int height = binary_image.rows;
        int width = binary_image.cols;

        cv::Mat visited = cv::Mat::zeros(binary_image.size(), CV_8U);
        std::vector<std::vector<cv::Point>> contours;

        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                if (is_edge(binary_image, y, x) && visited.at<uchar>(y, x) == 0) {
                    std::vector<cv::Point> contour = trace_contour(binary_image, visited, y, x);
                    if (contour.size() < 3)
                        continue;

                    cv::RotatedRect rect = cv::minAreaRect(contour);
                    float angle = std::abs(std::fmod(rect.angle, 180.0f));
                    if (angle > 90.0f) angle = 90.0f - angle;

                    if (std::abs(angle - 45.0f) <= angle_tolerance) {
                        contours.push_back(contour);
                    }
                }
            }
        }
        return contours;
    }

    std::tuple<cv::Point2f, float, float, float> min_area_rect(const std::vector<cv::Point>& contour) {
        if (contour.size() < 3)
            throw std::invalid_argument("Contour must have at least 3 points");

        // Compute convex hull
        std::vector<int> hull_indices;
        cv::convexHull(contour, hull_indices);

        std::vector<cv::Point> hull_points;
        for (auto idx : hull_indices)
            hull_points.push_back(contour[idx]);

        double min_area = std::numeric_limits<double>::max();
        cv::Point2f best_center;
        float best_width = 0, best_height = 0, best_angle = 0;

        // Rotating calipers style search
        for (size_t i = 0; i < hull_points.size(); ++i) {
            cv::Point2f p1 = hull_points[i];
            cv::Point2f p2 = hull_points[(i + 1) % hull_points.size()];
            cv::Point2f edge = p2 - p1;
            float angle = -std::atan2(edge.y, edge.x) * 180.0f / CV_PI;

            // Rotate hull points by -angle
            std::vector<cv::Point2f> rotated_points;
            cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point2f(0, 0), angle, 1.0);
            cv::transform(hull_points, rotated_points, rot_mat);

            // Get bounding rect in rotated frame
            float min_x = std::numeric_limits<float>::max();
            float min_y = std::numeric_limits<float>::max();
            float max_x = std::numeric_limits<float>::lowest();
            float max_y = std::numeric_limits<float>::lowest();

            for (const auto& pt : rotated_points) {
                if (pt.x < min_x) min_x = pt.x;
                if (pt.x > max_x) max_x = pt.x;
                if (pt.y < min_y) min_y = pt.y;
                if (pt.y > max_y) max_y = pt.y;
            }
            float width = max_x - min_x;
            float height = max_y - min_y;
            double area = width * height;

            if (area < min_area) {
                min_area = area;
                cv::Point2f center_rotated((min_x + max_x) / 2, (min_y + max_y) / 2);

                // Inverse rotate center to original coordinate frame
                cv::Mat inv_rot_mat = cv::getRotationMatrix2D(cv::Point2f(0, 0), -angle, 1.0);
                std::vector<cv::Point2f> center_vec = {center_rotated};
                std::vector<cv::Point2f> center_original;
                cv::transform(center_vec, center_original, inv_rot_mat);

                best_center = center_original[0];
                best_width = width;
                best_height = height;
                best_angle = angle;
            }
        }

        return {best_center, best_height, best_width, best_angle};
    }

    std::vector<cv::Point2f> rotate_points(const std::vector<cv::Point2f>& points, float angle) {
        cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point2f(0, 0), angle, 1.0);
        std::vector<cv::Point2f> rotated_points;
        cv::transform(points, rotated_points, rot_mat);
        return rotated_points;
    }

    std::vector<cv::Point2f> get_rectangle_corners(const cv::Point2f& center, float width, float height, float angle) {
        float w = width / 2.0f;
        float h = height / 2.0f;

        std::vector<cv::Point2f> corners = {
            {-w, -h}, {w, -h}, {w, h}, {-w, h}
        };

        auto rotated = rotate_points(corners, angle);
        for (auto& p : rotated)
            p += center;
        return rotated;
    }
}
