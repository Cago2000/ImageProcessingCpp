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

        //cv::imshow("mask", mask);
        //cv::waitKey(0);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (static_cast<int>(mask.at<uint8_t>(y, x)) == 255 && labels.at<int>(y, x) == 0) {
                    std::stack<cv::Point> stack;
                    std::vector<cv::Point> blob;

                    stack.emplace(x, y);

                    while (!stack.empty()) {
                        cv::Point pt = stack.top();
                        stack.pop();

                        int cx = pt.x;
                        int cy = pt.y;

                        if (cx < 0 || cx >= width || cy < 0 || cy >= height)
                            continue;

                        if(static_cast<int>(mask.at<uint8_t>(cy, cx)) == 255 && labels.at<int>(cy, cx) == 0) {
                            labels.at<int>(cy, cx) = label;
                            blob.push_back(pt);

                            stack.emplace(cx + 1, cy);
                            stack.emplace(cx - 1, cy);
                            stack.emplace(cx, cy + 1);
                            stack.emplace(cx, cy - 1);
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
