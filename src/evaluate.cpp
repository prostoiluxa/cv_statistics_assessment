#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <methods.h>

using json = nlohmann::json;

cv::Mat createCollageMask()
{
    const int cell_size = 256;
    const int roi_size = 228;
    const int border = (cell_size - roi_size) / 2; // (256-228)/2 = 14

    cv::Mat mask(1280, 1280, CV_8UC1, cv::Scalar(0));

    for (int row = 0; row < 5; row++)
    {
        for (int col = 0; col < 5; col++)
        {
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

json evaluateCollage(const cv::Mat &collage, const cv::Mat &mask)
{
    json j_result;
    std::vector<json> cells_array;

    const int cell_size = 256;
    const int roi_size = 228;
    const int border = (cell_size - roi_size) / 2;

    for (int row = 0; row < 5; row++)
    {
        for (int col = 0; col < 5; col++)
        {
            // Координаты ячейки
            cv::Rect cell_rect(col * cell_size, row * cell_size, cell_size, cell_size);

            // Координаты ROI внутри ячейки
            cv::Rect roi_rect(
                col * cell_size + border,
                row * cell_size + border,
                roi_size,
                roi_size);

            // Создаем маску для текущей ячейки
            cv::Mat cell_mask = mask(roi_rect).clone();

            // Вырезаем ROI изображения
            cv::Mat cell_roi = collage(roi_rect);

            // Преобразуем в float для вычислений
            cv::Mat cell_float;
            cell_roi.convertTo(cell_float, CV_32F);

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
    return j_result;
}

void processAllCollages()
{
    // Создаем общую маску (одинакова для всех коллажей)
    cv::Mat mask = createCollageMask();

    std::vector<int> distributions = {0, 1, 2};
    std::vector<double> snr_levels = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50};

    for (const auto &dist : distributions)
    {
        for (double snr_db : snr_levels)
        {
            // Загрузка коллажа
            std::string filename = "d" + std::to_string(dist) + "_snr" + std::to_string((int)snr_db) + "dB.tiff";
            std::string image_path = "../src/test_images/" + filename;
            cv::Mat collage = cv::imread(image_path, cv::IMREAD_UNCHANGED);

            if (collage.empty())
            {
                std::cerr << "Error loading: " << image_path << std::endl;
                continue;
            }

            // Обработка коллажа
            json evaluation = evaluateCollage(collage, mask);

            // Сохранение результатов
            std::string out_filename = "d" + std::to_string(dist) + "_snr" + std::to_string((int)snr_db) + "dB_eval.json";
            std::string out_path = "../src/evaluations/" + out_filename;
            std::ofstream out_file(out_path);
            out_file << std::setw(4) << evaluation << std::endl;

            std::cout << "Processed: " << out_path << std::endl;
        }
    }
}

// int main() {
//     processAllCollages();
//     return 0;
// }

int main(int argc, char **argv)
{
    if (argc != 3)
    {
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