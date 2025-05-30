#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <methods.h>

using json = nlohmann::json;


cv::Mat createCollageMask() {
    const int cell_size = 256;
    const int roi_size = 228;
    const int border = (cell_size - roi_size) / 2; // (256-228)/2 = 14
    
    cv::Mat mask(1280, 1280, CV_8UC1, cv::Scalar(0));
    
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 5; col++) {
            // Координаты ROI внутри ячейки
            int x_start = col * cell_size + border;
            int y_start = row * cell_size + border;
            
            // Создаем белый квадрат ROI
            cv::Rect roi_rect(x_start, y_start, roi_size, roi_size);
            mask(roi_rect).setTo(cv::Scalar(255));
        }
    }
    return mask;
}

void evaluateCollage(const std::string& eval_path, const cv::Mat& collage, const cv::Mat& mask) {
    json j_result;
    std::vector<json> cells_array;
    
    const int cell_size = 256;
    const int roi_size = 228;
    const int border = (cell_size - roi_size) / 2;

    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 5; col++) {
            // Координаты ячейки
            cv::Rect cell_rect(col * cell_size, row * cell_size, cell_size, cell_size);
            
            // Координаты ROI внутри ячейки
            cv::Rect roi_rect(
                col * cell_size + border,
                row * cell_size + border,
                roi_size,
                roi_size
            );
            
            cv::Mat cell_mask = mask(roi_rect).clone();
            
            cv::Mat cell_roi = collage(roi_rect);
            
            cv::Mat cell_float;
            cell_roi.convertTo(cell_float, CV_32F);
            
            // cv::imshow("cell_float",cell_float);
            // cv::imshow("cell_mask", cell_mask);

            // cv::waitKey(0);
            // Вычисляем статистики
            double skewness = getSkewnessValue(cell_float, cell_mask);
            double kurtosis = getKurtosisValue(cell_float, cell_mask);
            
            // Добавляем в JSON
            json j_cell;
            j_cell["row"] = row;
            j_cell["column"] = col;
            j_cell["evaluated_skewness"] = skewness;
            j_cell["evaluated_kurtosis"] = kurtosis;
            
            cells_array.push_back(j_cell);
        }
    }
    
    j_result["cells"] = cells_array;

    std::ofstream eval_file(eval_path);
    eval_file << std::setw(4) << j_result << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " [image_path] [eval_path] " << std::endl;
        return 1;
    }

    std::string image_path = argv[1];
    image_path = "../src/test_images/" + image_path;
    cv::Mat image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    std::string eval_path = argv[2];
    eval_path = "../src/evaluations/" + eval_path;

    evaluateCollage(eval_path, image, createCollageMask());
}