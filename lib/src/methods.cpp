#include "methods.h"

double getSkewnessValue(const cv::Mat& image, const cv::Mat& mask) {
    CV_Assert(image.channels() == 1); 
    CV_Assert(mask.type() == CV_8U);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(image, mean, stddev, mask);
    
    const double m = mean[0];
    const double sd = std::max(stddev[0], 1e-10); // Защита от деления на 0
    double sum3 = 0.0;
    int count = 0;

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            if (mask.at<uchar>(y, x)) {
                double val = image.at<float>(y, x);
                double diff = val - m;
                sum3 += diff * diff * diff; // (x-μ)^3
                count++;
            }
        }
    }
    
    return (sum3 / count) / (sd * sd * sd);
}

double getKurtosisValue(const cv::Mat& image, const cv::Mat& mask) {
    CV_Assert(image.channels() == 1);
    CV_Assert(mask.type() == CV_8U);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(image, mean, stddev, mask);
    
    const double m = mean[0];
    const double sd = std::max(stddev[0], 1e-10);
    double sum4 = 0.0;
    int count = 0;

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            if (mask.at<uchar>(y, x)) {
                double val = image.at<float>(y, x);
                double diff = val - m;
                sum4 += diff * diff * diff * diff; // (x-μ)^4
                count++;
            }
        }
    }
    
    return (sum4 / count) / (sd * sd * sd * sd) - 3.0;
}