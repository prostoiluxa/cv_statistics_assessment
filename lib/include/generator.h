#ifndef GENERATOR_H
#define GENERATOR_H

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <vector>
#include <random>

using json = nlohmann::json;

struct ElpsParams {
    int x;
    int y;
    int width;
    int height;
    double angle;
};

class ImageGenerator {
public:
    ImageGenerator(const std::string& config_path, int seed = -1);
    ImageGenerator(int seed = -1);
    static void generate_default_config(const std::string& path);
    cv::Mat generate_collage(const std::string& gt_path);
    void generateAll();

private:
    int distribution;
    double snr_db;
    unsigned int seed;
    std::mt19937 rng;

    void parseConfig(const json& config);
    cv::Mat applyGaussianNoise(const cv::Mat& image, double snr_db);
    cv::Mat generate_cell(double mean, double stddev);
    json createMetadata(int distribution, double snr_db);
    cv::Mat generateCollage1(int distribution);
    cv::Mat generate_cell1(int distribution, double mean, double stddev);
};

#endif