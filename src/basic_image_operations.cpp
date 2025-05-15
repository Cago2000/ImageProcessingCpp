#include "header/basic_image_operations.hpp"
#include "opencv2/opencv.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

namespace basic_ops {
    cv::Mat create_image(int width, int height, int channels, int gray_value) {
        return cv::Mat(height, width, CV_8UC(channels), cv::Scalar(gray_value));
    }

    cv::Mat create_image_with_gradient(int width, int height, int brightness) {
        cv::Mat image(height, width, CV_8UC3);
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) {
                int blue = static_cast<int>(brightness * (static_cast<float>(x) / width));
                int green = static_cast<int>(brightness * (static_cast<float>(y) / height));
                int red = static_cast<int>(brightness * ((x + y) / static_cast<float>(width + height)));
                image.at<cv::Vec3b>(y, x) = cv::Vec3b(blue, green, red);
            }
        }
        return image;
    }

    cv::Mat load_image(const std::string& image_path) {
        if (!fs::exists(image_path)) {
            std::cerr << "Error: File not found." << std::endl;
            return {};
        }
        cv::Mat img = cv::imread(image_path, cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            std::cerr << "Error: Unable to load image." << std::endl;
            return {};
        }
        if (img.channels() == 3 && fs::path(image_path).extension() == ".pgm") {
            cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        }
        std::cout << "Image loaded from " << image_path << std::endl;
        return img;
    }

    std::vector<cv::Mat> load_images(const std::string& folder_path, int amount) {
        std::vector<cv::Mat> images;
        for (const auto& entry : fs::directory_iterator(folder_path)) {
            std::string filename = entry.path().string();
            if (entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg" ||
                entry.path().extension() == ".png" || entry.path().extension() == ".ppm" ||
                entry.path().extension() == ".pgm") {
                cv::Mat img = load_image(filename);
                if (!img.empty()) images.push_back(img);
                if (images.size() >= static_cast<size_t>(amount)) break;
                }
        }
        return images;
    }

    void save_image(const cv::Mat& image, const std::string& save_path) {
        try {
            if (cv::imwrite(save_path, image)) {
                std::cout << "Image saved at " << save_path << std::endl;
            } else {
                throw std::runtime_error("cv::imwrite returned false");
            }
        } catch (const std::exception& e) {
            std::cerr << "Error saving image: " << e.what() << std::endl;
        }
    }

    void delete_image(const std::string& image_path) {
        if (fs::exists(image_path)) {
            fs::remove(image_path);
            std::cout << "Image deleted at " << image_path << std::endl;
        } else {
            std::cerr << "Error: File not found." << std::endl;
        }
    }

    void show_image(const cv::Mat& image, const std::string& title) {
        if (!image.empty()) {
            cv::imshow(title, image);
            cv::waitKey(0);
            cv::destroyAllWindows();
            std::cout << "Image displayed" << std::endl;
        } else {
            std::cerr << "Error: Cannot display an empty image." << std::endl;
        }
    }

    void create_ppm_image(int width, int height, const std::string& name, const std::string& file_format) {
        std::string file_path = "images/" + name + ".ppm";
        std::ofstream file(file_path, file_format == "ascii" ? std::ios::out : std::ios::binary);

        std::vector<std::tuple<uint8_t, uint8_t, uint8_t>> pixels = {
            {255, 0, 0}, {0, 255, 0}, {0, 0, 255},
            {255, 255, 0}, {255, 255, 255}, {0, 0, 0}
        };

        width = 3;
        height = 2;

        if (file_format == "ascii") {
            file << "P3\n" << width << " " << height << "\n255\n";
            for (const auto& [r, g, b] : pixels) {
                file << static_cast<int>(r) << " " << static_cast<int>(g) << " " << static_cast<int>(b) << " ";
            }
        } else {
            file << "P6\n" << width << " " << height << "\n255\n";
            for (const auto& [r, g, b] : pixels) {
                file.put(r).put(g).put(b);
            }
        }
    }

    cv::Mat load_ppm_image(const std::string& image_path) {
        std::ifstream file(image_path, std::ios::binary);
        std::string magic_number;
        file >> magic_number;

        if (magic_number != "P6") {
            throw std::runtime_error("Unsupported PPM format (only P6 is supported)");
        }

        int width, height, max_color;
        file >> width >> height >> max_color;
        file.ignore(); // Skip one byte (newline)

        if (max_color != 255) {
            throw std::runtime_error("Only max color value of 255 is supported");
        }

        std::vector<uint8_t> pixel_data(width * height * 3);
        file.read(reinterpret_cast<char*>(pixel_data.data()), pixel_data.size());

        cv::Mat image(height, width, CV_8UC3);
        std::memcpy(image.data, pixel_data.data(), pixel_data.size());
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        return image;
    }
}
