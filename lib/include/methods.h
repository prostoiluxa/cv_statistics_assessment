#ifndef METHODS_H
#define METHODS_H

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>

double getSkewnessValue(const cv::Mat& image, const cv::Mat& mask);
double getKurtosisValue(const cv::Mat& image, const cv::Mat& mask);

#endif