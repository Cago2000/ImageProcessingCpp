#include "header/color_detection.hpp"
#include <stack>
namespace cd {
    std::vector<std::vector<cv::Point>> get_blobs(cv::Mat mask) {
        CV_Assert(mask.type() == CV_8UC1);  // Expect a binary mask (1 channel)

        int label = 1;
        cv::Mat labels = cv::Mat::zeros(mask.size(), CV_32SC1);
        int height = mask.rows;
        int width = mask.cols;

        std::vector<std::vector<cv::Point>> blobs;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (mask.at<uint8_t>(y, x) == 1 && labels.at<int>(y, x) == 0) {
                    std::stack<cv::Point> stack;
                    std::vector<cv::Point> blob;

                    stack.push(cv::Point(x, y));

                    while (!stack.empty()) {
                        cv::Point pt = stack.top();
                        stack.pop();

                        int cx = pt.x;
                        int cy = pt.y;

                        if (cx < 0 || cx >= width || cy < 0 || cy >= height)
                            continue;

                        if (mask.at<uint8_t>(cy, cx) == 1 && labels.at<int>(cy, cx) == 0) {
                            labels.at<int>(cy, cx) = label;
                            blob.push_back(pt);

                            stack.push(cv::Point(cx + 1, cy));
                            stack.push(cv::Point(cx - 1, cy));
                            stack.push(cv::Point(cx, cy + 1));
                            stack.push(cv::Point(cx, cy - 1));
                        }
                    }

                    blobs.push_back(blob);
                    ++label;
                }
            }
        }
        return blobs;
    }

}
