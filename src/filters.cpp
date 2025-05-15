#include "header/filters.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <fftw3.h>

using namespace std;

namespace filters {
    cv::Mat grayScaleFilter(const cv::Mat& image) {
        if (image.channels() == 1) return image.clone();
        cv::Mat gray(image.rows, image.cols, CV_8UC1);
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
                gray.at<uchar>(y, x) = static_cast<uchar>(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);
            }
        }
        return gray;
    }

    cv::Mat blackWhiteFilter(const cv::Mat& image, int threshold) {
        CV_Assert(image.channels() == 1);
        cv::Mat result(image.size(), CV_8UC1);
        for (int y = 0; y < image.rows; ++y)
            for (int x = 0; x < image.cols; ++x)
                result.at<uchar>(y, x) = image.at<uchar>(y, x) >= threshold ? 255 : 0;
        return result;
    }

    cv::Mat blurFilter(const cv::Mat& image, int kernelDim, int kernelIntensity) {
        int pad = kernelDim / 2;
        cv::Mat padded;
        cv::copyMakeBorder(image, padded, pad, pad, pad, pad, cv::BORDER_CONSTANT, 0);
        cv::Mat output(image.size(), image.type());

        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                for (int c = 0; c < image.channels(); ++c) {
                    float sum = 0.0f;
                    for (int ky = 0; ky < kernelDim; ++ky) {
                        for (int kx = 0; kx < kernelDim; ++kx) {
                            sum += padded.at<cv::Vec3b>(y + ky, x + kx)[c];
                        }
                    }
                    output.at<cv::Vec3b>(y, x)[c] = static_cast<uchar>(sum / kernelIntensity);
                }
            }
        }
        return output;
    }

    cv::Mat sobelFilter(const cv::Mat& image, const std::string& mode, int intensity) {
        CV_Assert(image.channels() == 1);
        int sx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int sy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
        cv::Mat output = cv::Mat::zeros(image.size(), CV_8UC1);
        cv::Mat padded;
        cv::copyMakeBorder(image, padded, 1, 1, 1, 1, cv::BORDER_CONSTANT);

        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                int gx = 0, gy = 0;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) {
                        int val = padded.at<uchar>(y + i, x + j);
                        gx += sx[i][j] * val * intensity;
                        gy += sy[i][j] * val * intensity;
                    }

                if (mode == "vertical")
                    output.at<uchar>(y, x) = std::clamp(abs(gx), 0, 255);
                else if (mode == "horizontal")
                    output.at<uchar>(y, x) = std::clamp(abs(gy), 0, 255);
                else if (mode == "both")
                    output.at<uchar>(y, x) = std::clamp((int)std::sqrt(gx * gx + gy * gy), 0, 255);
            }
        }
        return output;
    }

    cv::Mat laplaceFilter(const cv::Mat& image, int intensity, int threshold) {
        CV_Assert(image.channels() == 1);
        int kernel[3][3] = {{0, -1, 0}, {-1, intensity, -1}, {0, -1, 0}};
        cv::Mat padded;
        cv::copyMakeBorder(image, padded, 1, 1, 1, 1, cv::BORDER_REPLICATE);
        cv::Mat output = cv::Mat::zeros(image.size(), CV_8UC1);

        for (int y = 0; y < image.rows; ++y)
            for (int x = 0; x < image.cols; ++x) {
                int sum = 0;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        sum += kernel[i][j] * padded.at<uchar>(y + i, x + j);
                sum = std::clamp(sum, 0, 255);
                output.at<uchar>(y, x) = threshold ? (sum >= threshold ? 255 : 0) : sum;
            }
        return output;
    }

    cv::Mat linearGrayScaling(cv::Mat image, float c1, float c2) {
        CV_Assert(image.channels() == 1);
        for (int y = 0; y < image.rows; ++y)
            for (int x = 0; x < image.cols; ++x) {
                float val = c2 * image.at<uchar>(y, x) + c1 * c2;
                image.at<uchar>(y, x) = std::min(255.0f, std::max(0.0f, val));
            }
        return image;
    }

    cv::Mat isodensityFilter(const cv::Mat& image, int degree) {
        CV_Assert(image.channels() == 1);
        cv::Scalar mean, stddev;
        cv::meanStdDev(image, mean, stddev);
        float mu = static_cast<float>(mean[0]);
        float sigma = static_cast<float>(stddev[0]);
        cv::Mat output(image.size(), CV_8UC1);

        for (int y = 0; y < image.rows; ++y)
            for (int x = 0; x < image.cols; ++x) {
                uchar val = image.at<uchar>(y, x);
                if (degree == 1) {
                    if (val < mu - sigma) output.at<uchar>(y, x) = 0;
                    else if (val > mu + sigma) output.at<uchar>(y, x) = 255;
                    else output.at<uchar>(y, x) = static_cast<uchar>(mu);
                } else if (degree == 2) {
                    output.at<uchar>(y, x) = val < mu ? 0 : 255;
                }
            }
        return output;
    }

    cv::Mat erosion(const cv::Mat& image, int dim) {
        int pad = dim / 2;
        cv::Mat padded, output(image.size(), image.type());
        cv::copyMakeBorder(image, padded, pad, pad, pad, pad, cv::BORDER_REPLICATE);
        for (int y = 0; y < image.rows; ++y)
            for (int x = 0; x < image.cols; ++x)
                for (int c = 0; c < image.channels(); ++c) {
                    uchar minVal = 255;
                    for (int i = 0; i < dim; ++i)
                        for (int j = 0; j < dim; ++j)
                            minVal = std::min(minVal, padded.at<cv::Vec3b>(y + i, x + j)[c]);
                    output.at<cv::Vec3b>(y, x)[c] = minVal;
                }
        return output;
    }

    cv::Mat dilation(const cv::Mat& image, int dim) {
        int pad = dim / 2;
        cv::Mat padded, output(image.size(), image.type());
        cv::copyMakeBorder(image, padded, pad, pad, pad, pad, cv::BORDER_REPLICATE);
        for (int y = 0; y < image.rows; ++y)
            for (int x = 0; x < image.cols; ++x)
                for (int c = 0; c < image.channels(); ++c) {
                    uchar maxVal = 0;
                    for (int i = 0; i < dim; ++i)
                        for (int j = 0; j < dim; ++j)
                            maxVal = std::max(maxVal, padded.at<cv::Vec3b>(y + i, x + j)[c]);
                    output.at<cv::Vec3b>(y, x)[c] = maxVal;
                }
        return output;
    }

    cv::Mat medianFilter(const cv::Mat& image, int dim) {
        int pad = dim / 2;
        cv::Mat padded, output(image.size(), image.type());
        cv::copyMakeBorder(image, padded, pad, pad, pad, pad, cv::BORDER_REPLICATE);

        for (int y = 0; y < image.rows; ++y)
            for (int x = 0; x < image.cols; ++x)
                for (int c = 0; c < image.channels(); ++c) {
                    std::vector<uchar> neighborhood;
                    for (int i = 0; i < dim; ++i)
                        for (int j = 0; j < dim; ++j)
                            neighborhood.push_back(padded.at<cv::Vec3b>(y + i, x + j)[c]);
                    std::nth_element(neighborhood.begin(), neighborhood.begin() + neighborhood.size() / 2, neighborhood.end());
                    output.at<cv::Vec3b>(y, x)[c] = neighborhood[neighborhood.size() / 2];
                }
        return output;
    }
    cv::Mat centerKernel(const cv::Mat& kernel, cv::Size targetSize) {
        cv::Mat padded = cv::Mat::zeros(targetSize, CV_32F);
        int cy = targetSize.height / 2 - kernel.rows / 2;
        int cx = targetSize.width / 2 - kernel.cols / 2;
        kernel.copyTo(padded(cv::Rect(cx, cy, kernel.cols, kernel.rows)));
        return padded;
    }

    cv::Mat sobelFilterFFT(const cv::Mat& inputGray, const string& mode, int intensity) {
        if (inputGray.channels() != 1)
            return cv::Mat();

        int height = inputGray.rows;
        int width = inputGray.cols;

        cv::Mat sobelX = (cv::Mat_<float>(3,3) <<
            -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1) * intensity;

        cv::Mat sobelY = (cv::Mat_<float>(3,3) <<
            -1, -2, -1,
             0,  0,  0,
             1,  2,  1) * intensity;

        cv::Mat inputFloat;
        inputGray.convertTo(inputFloat, CV_32F);

        // Prepare kernels
        cv::Mat kx = centerKernel(sobelX, inputGray.size());
        cv::Mat ky = centerKernel(sobelY, inputGray.size());

        // FFTW setup
        fftwf_complex *in, *outKx, *outKy, *outImg;
        fftwf_plan planKx, planKy, planImg;

        int N = height * width;
        in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
        outKx = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
        outKy = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
        outImg = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);

        auto matToFFTW = [&](const cv::Mat& src, fftwf_complex* dst) {
            for (int i = 0; i < src.rows; ++i)
                for (int j = 0; j < src.cols; ++j) {
                    dst[i * width + j][0] = src.at<float>(i, j);
                    dst[i * width + j][1] = 0.0f;
                }
        };

        auto fft = [&](fftwf_complex* data, fftwf_complex* result) {
            fftwf_plan p = fftwf_plan_dft_2d(height, width, data, result, FFTW_FORWARD, FFTW_ESTIMATE);
            fftwf_execute(p);
            fftwf_destroy_plan(p);
        };

        auto ifft = [&](fftwf_complex* data, cv::Mat& result) {
            fftwf_plan p = fftwf_plan_dft_2d(height, width, data, data, FFTW_BACKWARD, FFTW_ESTIMATE);
            fftwf_execute(p);
            result = cv::Mat::zeros(height, width, CV_32F);
            for (int i = 0; i < height; ++i)
                for (int j = 0; j < width; ++j)
                    result.at<float>(i, j) = sqrtf(data[i * width + j][0]*data[i * width + j][0] +
                                                   data[i * width + j][1]*data[i * width + j][1]) / (height * width);
            fftwf_destroy_plan(p);
        };

        matToFFTW(kx, in); fft(in, outKx);
        matToFFTW(ky, in); fft(in, outKy);
        matToFFTW(inputFloat, in); fft(in, outImg);

        cv::Mat result;
        if (mode == "vertical" || mode == "both") {
            fftwf_complex* outFiltered = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
            for (int i = 0; i < N; ++i) {
                outFiltered[i][0] = outImg[i][0] * outKx[i][0] - outImg[i][1] * outKx[i][1];
                outFiltered[i][1] = outImg[i][0] * outKx[i][1] + outImg[i][1] * outKx[i][0];
            }
            cv::Mat gx; ifft(outFiltered, gx);
            if (mode == "vertical")
                result = gx;
            fftwf_free(outFiltered);
        }
        if (mode == "horizontal" || mode == "both") {
            fftwf_complex* outFiltered = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
            for (int i = 0; i < N; ++i) {
                outFiltered[i][0] = outImg[i][0] * outKy[i][0] - outImg[i][1] * outKy[i][1];
                outFiltered[i][1] = outImg[i][0] * outKy[i][1] + outImg[i][1] * outKy[i][0];
            }
            cv::Mat gy; ifft(outFiltered, gy);
            if (mode == "horizontal")
                result = gy;
            else if (mode == "both") {
                cv::Mat gx;
                ifft(outImg, gx);
                result = gx.mul(gx) + gy.mul(gy);
                sqrt(result, result);
            }
            fftwf_free(outFiltered);
        }

        fftwf_free(in); fftwf_free(outKx); fftwf_free(outKy); fftwf_free(outImg);
        result.convertTo(result, CV_8U, 1.0, 0.0);
        return result;
    }

    // Median Filter using Sorted Window
    cv::Mat medianFilterSorted(const cv::Mat& src, int dim) {
        int pad = dim / 2;
        cv::Mat padded;
        copyMakeBorder(src, padded, pad, pad, pad, pad, cv::BORDER_REFLECT);

        cv::Mat dst = cv::Mat::zeros(src.size(), src.type());

        if (src.channels() == 1) {
            for (int y = 0; y < src.rows; ++y) {
                for (int x = 0; x < src.cols; ++x) {
                    vector<uchar> window;
                    for (int dy = -pad; dy <= pad; ++dy) {
                        for (int dx = -pad; dx <= pad; ++dx) {
                            window.push_back(padded.at<uchar>(y + pad + dy, x + pad + dx));
                        }
                    }
                    nth_element(window.begin(), window.begin() + window.size()/2, window.end());
                    dst.at<uchar>(y, x) = window[window.size()/2];
                }
            }
        } else {
            vector<cv::Mat> channels;
            split(src, channels);
            for (int c = 0; c < src.channels(); ++c)
                channels[c] = medianFilterSorted(channels[c], dim);
            merge(channels, dst);
        }

    return dst;
    }
}
