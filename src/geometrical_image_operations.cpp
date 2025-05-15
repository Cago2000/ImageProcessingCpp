#include "header/geometrical_image_operations.hpp"
#include <cmath>

// Nearest neighbor resize
namespace geo_ops {
    cv::Mat resize_image(const cv::Mat& image, int target_width, int target_height) {
        int width = image.cols;
        int height = image.rows;
        int channels = image.channels();

        cv::Mat resized(target_height, target_width, image.type());

        for (int m = 0; m < target_width; ++m) {
            for (int n = 0; n < target_height; ++n) {
                float x = (float)m / (target_width - 1) * (width - 1);
                float y = (float)n / (target_height - 1) * (height - 1);
                int ix = static_cast<int>(y);
                int jx = static_cast<int>(x);

                if (channels == 1) {
                    resized.at<uchar>(n, m) = image.at<uchar>(ix, jx);
                } else {
                    for (int c = 0; c < channels; ++c) {
                        resized.at<cv::Vec3b>(n, m)[c] = image.at<cv::Vec3b>(ix, jx)[c];
                    }
                }
            }
        }
        return resized;
    }

    cv::Mat rotate_image(const cv::Mat& image, int degree) {
        if (degree % 360 == 0) {
            return image.clone();
        }

        int width = image.cols;
        int height = image.rows;
        int channels = image.channels();

        double rad = degree * M_PI / 180.0;

        double cos_theta = std::abs(std::cos(rad));
        double sin_theta = std::abs(std::sin(rad));

        int new_width = static_cast<int>(width * cos_theta + height * sin_theta);
        int new_height = static_cast<int>(width * sin_theta + height * cos_theta);

        cv::Mat output(new_height, new_width, image.type(), cv::Scalar::all(0));

        double original_cx = width / 2.0;
        double original_cy = height / 2.0;
        double new_cx = new_width / 2.0;
        double new_cy = new_height / 2.0;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                double x_shifted = x - original_cx;
                double y_shifted = y - original_cy;

                int new_x = static_cast<int>(std::round(std::cos(rad) * x_shifted - std::sin(rad) * y_shifted + new_cx));
                int new_y = static_cast<int>(std::round(std::sin(rad) * x_shifted + std::cos(rad) * y_shifted + new_cy));

                if (new_x >= 0 && new_x < new_width && new_y >= 0 && new_y < new_height) {
                    if (channels == 1) {
                        output.at<uchar>(new_y, new_x) = image.at<uchar>(y, x);
                    } else {
                        output.at<cv::Vec3b>(new_y, new_x) = image.at<cv::Vec3b>(y, x);
                    }
                }
            }
        }

        // Fill black holes by averaging neighbors
        for (int y = 0; y < new_height; ++y) {
            for (int x = 0; x < new_width; ++x) {
                bool is_black;
                if (channels == 1) {
                    is_black = (output.at<uchar>(y, x) == 0);
                } else {
                    cv::Vec3b pix = output.at<cv::Vec3b>(y, x);
                    is_black = (pix[0] == 0 && pix[1] == 0 && pix[2] == 0);
                }
                if (!is_black) continue;

                // Gather neighbors that are not black
                std::vector<cv::Vec3d> neighbors;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx < 0 || nx >= new_width || ny < 0 || ny >= new_height) continue;

                        bool neighbor_black;
                        if (channels == 1) {
                            neighbor_black = (output.at<uchar>(ny, nx) == 0);
                        } else {
                            cv::Vec3b npix = output.at<cv::Vec3b>(ny, nx);
                            neighbor_black = (npix[0] == 0 && npix[1] == 0 && npix[2] == 0);
                        }
                        if (!neighbor_black) {
                            if (channels == 1) {
                                neighbors.emplace_back(output.at<uchar>(ny, nx), 0, 0);
                            } else {
                                cv::Vec3b npix = output.at<cv::Vec3b>(ny, nx);
                                neighbors.emplace_back(npix[0], npix[1], npix[2]);
                            }
                        }
                    }
                }

                if (!neighbors.empty()) {
                    if (channels == 1) {
                        double avg = 0.0;
                        for (auto& n : neighbors) avg += n[0];
                        avg /= neighbors.size();
                        output.at<uchar>(y, x) = static_cast<uchar>(std::round(avg));
                    } else {
                        cv::Vec3d avg(0, 0, 0);
                        for (auto& n : neighbors) avg += n;
                        avg /= static_cast<double>(neighbors.size());
                        output.at<cv::Vec3b>(y, x) = cv::Vec3b(
                            static_cast<uchar>(std::round(avg[0])),
                            static_cast<uchar>(std::round(avg[1])),
                            static_cast<uchar>(std::round(avg[2]))
                        );
                    }
                }
            }
        }

        return output;
    }

    cv::Mat mirror_image(const cv::Mat& image, const std::string& mode) {
        int width = image.cols;
        int height = image.rows;
        int channels = image.channels();

        cv::Mat output(height, width, image.type());

        if (mode == "vertical") {
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    if (channels == 1) {
                        output.at<uchar>(i, j) = image.at<uchar>(i, width - j - 1);
                    } else {
                        output.at<cv::Vec3b>(i, j) = image.at<cv::Vec3b>(i, width - j - 1);
                    }
                }
            }
        } else if (mode == "horizontal") {
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    if (channels == 1) {
                        output.at<uchar>(i, j) = image.at<uchar>(height - i - 1, j);
                    } else {
                        output.at<cv::Vec3b>(i, j) = image.at<cv::Vec3b>(height - i - 1, j);
                    }
                }
            }
        } else {
            // Unknown mode: return copy of original image
            output = image.clone();
        }

        return output;
    }
}
