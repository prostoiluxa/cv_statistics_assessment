#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

const double EPSILON = 1e-6;

struct ErrorMetrics
{
    double mean_skewness_error;
    double mean_kurtosis_error;
    std::vector<double> skewness_errors;
    std::vector<double> kurtosis_errors;
};

struct CollageErrorMetrics
{
    int distribution;
    double snr_db;
    double mean_skewness_error;
    double mean_kurtosis_error;
    std::vector<double> skewness_errors;
    std::vector<double> kurtosis_errors;
};

struct DistributionMetrics
{
    std::vector<double> snr_levels;
    std::vector<double> skewness_errors;
    std::vector<double> kurtosis_errors;
};

double relativeError(double groundTruth, double evaluated)
{
    if (fabs(groundTruth) < EPSILON)
    {
        return fabs(evaluated); // Для нулевых значений используем абсолютную ошибку
    }
    return fabs((evaluated - groundTruth) / groundTruth);
};

ErrorMetrics compareJsonFiles(const std::string &gt_path, const std::string &eval_path)
{
    // Загрузка файлов
    std::ifstream gt_file(gt_path);
    std::ifstream eval_file(eval_path);
    json gt_data = json::parse(gt_file);
    json eval_data = json::parse(eval_file);

    ErrorMetrics metrics;
    double total_skew_error = 0.0;
    double total_kurt_error = 0.0;
    int cell_count = 0;

    // Проход по всем ячейкам
    for (const auto &eval_cell : eval_data["cells"])
    {
        int row = eval_cell["row"];
        int col = eval_cell["column"];

        // Поиск соответствующей ячейки в ground truth
        const auto &gt_row = gt_data["cells"][row];
        const auto &gt_cell = gt_row[col];

        // Получение значений
        double gt_skew = gt_cell["theoretical_skewness"];
        double gt_kurt = gt_cell["theoretical_kurtosis"];
        double eval_skew = eval_cell["evaluated_skewness"];
        double eval_kurt = eval_cell["evaluated_kurtosis"];

        // Расчет ошибок
        double skew_error = relativeError(gt_skew, eval_skew);
        double kurt_error = relativeError(gt_kurt, eval_kurt);

        // Сохранение ошибок
        metrics.skewness_errors.push_back(skew_error);
        metrics.kurtosis_errors.push_back(kurt_error);

        total_skew_error += skew_error;
        total_kurt_error += kurt_error;
        cell_count++;
    }

    // Расчет средних ошибок
    metrics.mean_skewness_error = total_skew_error / cell_count;
    metrics.mean_kurtosis_error = total_kurt_error / cell_count;

    return metrics;
}

std::vector<CollageErrorMetrics> collectAllErrorMetrics(
    const std::vector<int> &distributions,
    const std::vector<double> &snr_levels)
{

    std::vector<CollageErrorMetrics> all_metrics;

    for (const auto &dist : distributions)
    {
        for (double snr_db : snr_levels)
        {
            // Формирование путей к файлам
            std::string gt_filename = "d" + std::to_string(dist) + "_snr" + std::to_string((int)snr_db) + "dB.json";
            std::string gt_path = "../src/gt/" + gt_filename;
            std::string eval_filename = "d" + std::to_string(dist) + "_snr" + std::to_string((int)snr_db) + "dB_eval.json";
            std::string eval_path = "../src/evaluations/" + eval_filename;

            // Проверка существования файлов
            if (!fs::exists(gt_path))
            {
                std::cerr << "Warning: GT file not found - " << gt_path << std::endl;
                continue;
            }
            if (!fs::exists(eval_path))
            {
                std::cerr << "Warning: Eval file not found - " << eval_path << std::endl;
                continue;
            }

            // Сравнение файлов
            ErrorMetrics metrics = compareJsonFiles(gt_path, eval_path);

            // Сохранение результатов с идентификаторами
            CollageErrorMetrics collage_metrics;
            collage_metrics.distribution = dist;
            collage_metrics.snr_db = snr_db;
            collage_metrics.mean_skewness_error = metrics.mean_skewness_error;
            collage_metrics.mean_kurtosis_error = metrics.mean_kurtosis_error;
            collage_metrics.skewness_errors = std::move(metrics.skewness_errors);
            collage_metrics.kurtosis_errors = std::move(metrics.kurtosis_errors);

            all_metrics.push_back(std::move(collage_metrics));
        }
    }

    return all_metrics;
};

void plotErrorMetrics(const std::map<int, DistributionMetrics> &metrics_map)
{
    // Определение цветов для каждого распределения
    const std::map<int, cv::Scalar> distribution_colors = {
        {0, cv::Scalar(0, 0, 255)},   // Красный
        {1, cv::Scalar(0, 165, 255)}, // Оранжевый
        {2, cv::Scalar(0, 255, 0)}    // Зеленый
    };

    // Параметры графика
    const int WIDTH = 1200;
    const int HEIGHT = 800;
    const int MARGIN = 100;
    const int GRAPH_HEIGHT = HEIGHT - 2 * MARGIN;
    const int GRAPH_WIDTH = WIDTH - 2 * MARGIN;

    // Создание изображения для графика
    cv::Mat plot_image(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(255, 255, 255));

    // Находим максимальные значения для масштабирования
    double max_snr = 50.0;
    double max_error = 0.0;

    for (const auto &[dist, metrics] : metrics_map)
    {
        for (double err : metrics.skewness_errors)
        {
            if (err > max_error)
                max_error = err;
        }
        for (double err : metrics.kurtosis_errors)
        {
            if (err > max_error)
                max_error = err;
        }
    }

    // Добавляем запас 10% сверху
    max_error *= 1.1;

    // Функция преобразования значений в координаты
    auto toPlotCoords = [&](double snr, double error) -> cv::Point
    {
        int x = MARGIN + static_cast<int>((snr / max_snr) * GRAPH_WIDTH);
        int y = HEIGHT - MARGIN - static_cast<int>((error / max_error) * GRAPH_HEIGHT);
        return cv::Point(x, y);
    };

    // Отрисовка осей
    cv::line(plot_image,
             cv::Point(MARGIN, HEIGHT - MARGIN),
             cv::Point(WIDTH - MARGIN, HEIGHT - MARGIN),
             cv::Scalar(0, 0, 0), 2);

    cv::line(plot_image,
             cv::Point(MARGIN, HEIGHT - MARGIN),
             cv::Point(MARGIN, MARGIN),
             cv::Scalar(0, 0, 0), 2);

    // Подписи осей
    cv::putText(plot_image, "SNR (dB)",
                cv::Point(WIDTH / 2 - 50, HEIGHT - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);


    // Разметка оси X (SNR)
    for (int snr = 0; snr <= 50; snr += 10)
    {
        int x = MARGIN + static_cast<int>(static_cast<double>(snr) / max_snr) * GRAPH_WIDTH;
        cv::line(plot_image, cv::Point(x, HEIGHT - MARGIN - 5),
                 cv::Point(x, HEIGHT - MARGIN + 5), cv::Scalar(0, 0, 0), 2);
        cv::putText(plot_image, std::to_string(snr),
                    cv::Point(x - 10, HEIGHT - MARGIN + 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    }

    // Разметка оси Y (Error)
    for (double error = 0.0; error <= max_error; error += 0.1)
    {
        int y = HEIGHT - MARGIN - static_cast<int>((error / max_error) * GRAPH_HEIGHT);
        cv::line(plot_image, cv::Point(MARGIN - 5, y),
                 cv::Point(MARGIN + 5, y), cv::Scalar(0, 0, 0), 1);

        std::string label = std::to_string(static_cast<int>(error * 100)) + "%";
        cv::putText(plot_image, label,
                    cv::Point(MARGIN - 90, y + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    }

    // Отрисовка графиков для каждого распределения
    for (const auto &[dist, metrics] : metrics_map)
    {
        if (dist != 2) continue;
        cv::Scalar color = distribution_colors.at(dist);
        std::vector<cv::Point> skew_points, kurt_points;

        // Собираем точки для графиков
        for (size_t i = 0; i < metrics.snr_levels.size(); ++i)
        {
            skew_points.push_back(toPlotCoords(metrics.snr_levels[i], metrics.skewness_errors[i]));
            kurt_points.push_back(toPlotCoords(metrics.snr_levels[i], metrics.kurtosis_errors[i]));
        }

        // Рисуем линии
        for (size_t i = 0; i < skew_points.size() - 1; ++i)
        {
            cv::line(plot_image, skew_points[i], skew_points[i + 1],
                     color, 2, cv::LINE_AA);
            cv::line(plot_image, kurt_points[i], kurt_points[i + 1],
                     cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
        }

        // Рисуем точки
        for (const auto &pt : skew_points)
        {
            cv::circle(plot_image, pt, 5, color, -1);
        }
        for (const auto &pt : kurt_points)
        {
            cv::circle(plot_image, pt, 5, cv::Scalar(255, 0, 0), -1);
        }
    }

    // Легенда
    cv::rectangle(plot_image, cv::Rect(WIDTH - 300, 50, 250, 150), cv::Scalar(200, 200, 200), -1);
    cv::rectangle(plot_image, cv::Rect(WIDTH - 300, 50, 250, 150), cv::Scalar(0, 0, 0), 2);

    int y_pos = 90;
    for (const auto &[dist, color] : distribution_colors)
    {
        if (dist != 2) continue;
        std::string text;
        if (dist == 0)
            text = "normal skewness";
        if (dist == 1)
            text = "uniform skewness";
        if (dist == 2)
            text = "exp skewness";
        cv::line(plot_image, cv::Point(WIDTH - 280, y_pos),
                 cv::Point(WIDTH - 240, y_pos), color, 2);
        cv::putText(plot_image, text,
                    cv::Point(WIDTH - 220, y_pos + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        y_pos += 30;
    }

    cv::line(plot_image, cv::Point(WIDTH - 280, y_pos),
             cv::Point(WIDTH - 240, y_pos), cv::Scalar(255, 0, 0), 2);
    cv::putText(plot_image, "kurtosis (all)",
                cv::Point(WIDTH - 220, y_pos + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

    // Заголовок
    cv::putText(plot_image, "Dependence of Error on SNR",
                cv::Point(WIDTH / 2 - 200, 40),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);

    // Сохраняем результат
    cv::imwrite("../src/assessment/error_vs_snr_plot.png", plot_image);
    cv::imshow("Error vs SNR", plot_image);
    cv::waitKey(0);
}

void analyzeErrorMetrics(const std::vector<CollageErrorMetrics> &all_metrics)
{
    // Группируем метрики по типам распределений
    std::map<int, DistributionMetrics> metrics_map;
    std::vector<double> snr_levels = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50};

    for (const int dist : {0, 1, 2})
    {
        metrics_map[dist] = DistributionMetrics{
            snr_levels,
            std::vector<double>(snr_levels.size(), 0.0),
            std::vector<double>(snr_levels.size(), 0.0)};
    }

    for (const auto &metrics : all_metrics)
    {
        auto it = std::find(snr_levels.begin(), snr_levels.end(), metrics.snr_db);
        if (it == snr_levels.end())
            continue;
        int idx = std::distance(snr_levels.begin(), it);

        metrics_map[metrics.distribution].skewness_errors[idx] = metrics.mean_skewness_error;
        metrics_map[metrics.distribution].kurtosis_errors[idx] = metrics.mean_kurtosis_error;
    }

    // Строим график
    plotErrorMetrics(metrics_map);

    // Дополнительный анализ
    std::cout << "\n===== Анализ зависимости ошибок от SNR =====";
    for (const auto &[dist, metrics] : metrics_map)
    {
        std::cout << "\n\n--- " << dist << " ---";
        std::cout << "\nСредняя ошибка при низком SNR (0-10dB): "
                  << (*std::max_element(metrics.skewness_errors.begin(), metrics.skewness_errors.begin() + 3)) * 100 << "% (skew), "
                  << (*std::max_element(metrics.kurtosis_errors.begin(), metrics.kurtosis_errors.begin() + 3)) * 100 << "% (kurt)";

        std::cout << "\nСредняя ошибка при высоком SNR (40-50dB): "
                  << (*std::min_element(metrics.skewness_errors.end() - 3, metrics.skewness_errors.end())) * 100 << "% (skew), "
                  << (*std::min_element(metrics.kurtosis_errors.end() - 3, metrics.kurtosis_errors.end())) * 100 << "% (kurt)";

        // Находим SNR, где ошибка становится приемлемой (<10%)
        auto skew_acceptable = std::find_if(metrics.skewness_errors.begin(), metrics.skewness_errors.end(),
                                            [](double err)
                                            { return err < 0.05; });
        auto kurt_acceptable = std::find_if(metrics.kurtosis_errors.begin(), metrics.kurtosis_errors.end(),
                                            [](double err)
                                            { return err < 0.05; });

        if (skew_acceptable != metrics.skewness_errors.end())
        {
            int idx = std::distance(metrics.skewness_errors.begin(), skew_acceptable);
            std::cout << "\nАсимметрия становится точной (<5%) при SNR > " << snr_levels[idx] << "dB";
        }

        if (kurt_acceptable != metrics.kurtosis_errors.end())
        {
            int idx = std::distance(metrics.kurtosis_errors.begin(), kurt_acceptable);
            std::cout << "\nКуртозис становится точным (<5%) при SNR > " << snr_levels[idx] << "dB";
        }
    }
}

void exportToCSV(const std::vector<CollageErrorMetrics> &metrics,
                 const std::string &filename,
                 bool includeHeader = true)
{

    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Настройка точности вывода
    file << std::fixed << std::setprecision(6);

    // Заголовок CSV
    if (includeHeader)
    {
        file << "Distribution,SNR_dB,MeanSkewnessError,MeanKurtosisError\n";
    }

    // Запись данных
    for (const auto &m : metrics)
    {
        file << m.distribution << ","
             << m.snr_db << ","
             << m.mean_skewness_error << ","
             << m.mean_kurtosis_error << "\n";
    }

    std::cout << "Exported " << metrics.size() << " records to " << filename << std::endl;
}

int main()
{
    std::vector<int> distributions = {0, 1, 2};
    std::vector<double> snr_levels = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50};

    std::vector<CollageErrorMetrics> all_metrics = collectAllErrorMetrics(distributions, snr_levels);
    analyzeErrorMetrics(all_metrics);
    exportToCSV(all_metrics, "../src/assessment/metrics.csv");

    return 0;
}