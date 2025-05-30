#include "generator.h"
#include <cmath>
#include <fstream>

ImageGenerator::ImageGenerator(const std::string &config_path, int seed)
{
    std::ifstream f(config_path);
    json config;
    f >> config;
    parseConfig(config);
    if (seed == -1)
    {
        rng.seed(this->seed);
    }
    else
    {
        this->seed = static_cast<unsigned>(seed);
        rng.seed(this->seed);
    }
}

ImageGenerator::ImageGenerator(int seed)
{
    if (seed == -1)
    {
        rng.seed(this->seed);
    }
    else
    {
        this->seed = static_cast<unsigned>(seed);
        rng.seed(this->seed);
    }
}

cv::Mat ImageGenerator::generate_collage(const std::string &gt_path)
{
    json gt_json;
    gt_json["distribution"] = distribution;
    gt_json["snr_db"] = snr_db;

    std::vector<double> means = {44.0, 88.0, 132.0, 176.0, 220.0};
    std::vector<double> stddevs = {0.5, 1.0, 1.5, 2.0, 2.5};

    const int cell_size = 256;
    cv::Mat collage(means.size() * cell_size, stddevs.size() * cell_size, CV_32FC1);

    json objects = json::array();

    for (int row = 0; row < stddevs.size(); ++row)
    {
        for (int col = 0; col < means.size(); ++col)
        {

            cv::Mat cell = generate_cell(stddevs[row], means[col]);

            cv::Rect roi(col * cell_size, row * cell_size, cell_size, cell_size);
            cell.copyTo(collage(roi));

            // Теоретические значения для распределений
            auto getTheoretical = [](const int dist) -> std::pair<double, double>
            {
                if (dist == 0)
                    return {0.0, 0.0}; // skewness, kurtosis
                if (dist == 1)
                    return {0.0, -1.2}; // excess kurtosis
                if (dist == 2)
                    return {2.0, 6.0}; // excess kurtosis
                return {0.0, 0.0};
            };

            auto [theor_skew, theor_kurt] = getTheoretical(distribution);

            json obj;
            obj["pic_coordinates"]["row"] = row;
            obj["pic_coordinates"]["col"] = col;
            obj["theoretical_skewness"] = theor_skew;
            obj["theoretical_kurtosis"] = theor_kurt;
            obj["mean"] = means[col];
            obj["std"] = stddevs[row];

            objects.push_back(obj);
        }
    }

    // cv::Mat noisy_collage = applyGaussianNoise(collage, snr_db);

    gt_json["cells"] = objects;

    std::ofstream gt_file(gt_path);
    gt_file << std::setw(4) << gt_json << std::endl;
    return collage;
}

cv::Mat ImageGenerator::generateCollage1(int distribution)
{
    cv::Mat collage(1280, 1280, CV_32FC1); // 5*256 = 1280

    std::vector<double> means = {44.0, 88.0, 132.0, 176.0, 220.0};
    std::vector<double> stddevs = {0.5, 1.0, 1.5, 2.0, 2.5};

    for (int row = 0; row < 5; row++)
    {
        for (int col = 0; col < 5; col++)
        {
            cv::Mat cell = generate_cell1(distribution, means[row], stddevs[col]);
            cell.copyTo(collage(cv::Rect(col * 256, row * 256, 256, 256)));
        }
    }

    return collage;
}

cv::Mat ImageGenerator::generate_cell(double mean, double stddev)
{
    cv::Mat cell(256, 256, CV_32FC1, cv::Scalar(0.0f)); // Черный фон

    cv::Mat roi = cell(cv::Rect(14, 14, 228, 228));

    if (distribution == 0)
    {
        cv::randn(roi, mean, stddev);
    }
    else if (distribution == 1)
    {
        double a = mean - std::sqrt(3.0) * stddev;
        double b = mean + std::sqrt(3.0) * stddev;
        cv::randu(roi, cv::Scalar(a), cv::Scalar(b));
    }
    else if (distribution == 2)
    {
        std::exponential_distribution<double> exp_dist(1.0 / mean);
        for (int y = 0; y < roi.rows; y++)
        {
            float *ptr = roi.ptr<float>(y);
            for (int x = 0; x < roi.cols; x++)
            {
                ptr[x] = exp_dist(rng);
            }
        }
    }

    double signal_power = stddev * stddev;
    double snr_linear = std::pow(10.0, snr_db / 10.0);
    double noise_power = signal_power / snr_linear;
    double noise_stddev = std::sqrt(noise_power);

    cv::Mat noise(cell.size(), CV_32FC1);
    cv::randn(noise, 128.0, noise_stddev);

    return cell + noise;
}

cv::Mat ImageGenerator::generate_cell1(int distribution, double mean, double stddev)
{
    cv::Mat cell(256, 256, CV_32FC1, cv::Scalar(0.0f)); // Черный фон

    // Внутренний квадрат (228x228)
    cv::Mat roi = cell(cv::Rect(14, 14, 228, 228));

    if (distribution == 0)
    {
        cv::randn(roi, mean, stddev);
    }
    else if (distribution == 1)
    {
        double a = mean - std::sqrt(3.0) * stddev;
        double b = mean + std::sqrt(3.0) * stddev;
        cv::randu(roi, cv::Scalar(a), cv::Scalar(b));
    }
    else if (distribution == 2)
    {
        std::exponential_distribution<double> exp_dist(1.0 / mean);
        for (int y = 0; y < roi.rows; y++)
        {
            float *ptr = roi.ptr<float>(y);
            for (int x = 0; x < roi.cols; x++)
            {
                ptr[x] = exp_dist(rng);
            }
        }
    }

    return cell;
}

cv::Mat ImageGenerator::applyGaussianNoise(const cv::Mat &image, double snr_db)
{
    CV_Assert(image.type() == CV_32FC1);

    cv::Scalar mean, stddev;
    cv::meanStdDev(image, mean, stddev);
    double signal_power = stddev.val[0] * stddev.val[0];

    double snr_linear = std::pow(10.0, snr_db / 10.0);
    double noise_power = signal_power / snr_linear;
    double noise_stddev = std::sqrt(noise_power);

    cv::Mat noise(image.size(), CV_32FC1);
    cv::randn(noise, 0.0, noise_stddev);

    return image + noise;
}

json ImageGenerator::createMetadata(int distribution, double snr_db)
{
    json j;
    j["distribution"] = distribution;
    j["snr_db"] = snr_db;

    std::vector<double> means = {44.0, 88.0, 132.0, 176.0, 220.0};
    std::vector<double> stddevs = {0.5, 1.0, 1.5, 2.0, 2.5};

    // Теоретические значения для распределений
    auto getTheoretical = [](const int dist) -> std::pair<double, double>
    {
        if (dist == 0)
            return {0.0, 0.0}; // skewness, kurtosis
        if (dist == 1)
            return {0.0, -1.2}; // excess kurtosis
        if (dist == 2)
            return {2.0, 6.0}; // excess kurtosis
        return {0.0, 0.0};
    };

    auto [theor_skew, theor_kurt] = getTheoretical(distribution);

    for (int row = 0; row < 5; row++)
    {
        json j_row;
        for (int col = 0; col < 5; col++)
        {
            json j_cell;
            j_cell["mean"] = means[row];
            j_cell["stddev"] = stddevs[col];
            j_cell["theoretical_skewness"] = theor_skew;
            j_cell["theoretical_kurtosis"] = theor_kurt;
            j_row.push_back(j_cell);
        }
        j["cells"].push_back(j_row);
    }

    return j;
}

void ImageGenerator::generateAll()
{
    std::vector<int> distributions = {0, 1, 2};
    std::vector<double> snr_levels = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50};

    for (const auto &dist : distributions)
    {
        cv::Mat clean_collage = generateCollage1(dist);

        for (double snr_db : snr_levels)
        {
            // Генерация изображения с шумом
            cv::Mat noisy_collage = applyGaussianNoise(clean_collage, snr_db);

            // Сохранение изображения
            std::string filename = "d" + std::to_string(dist) + "_snr" + std::to_string((int)snr_db) + "dB.tiff";
            std::string image_path = "../src/test_images/" + filename;
            cv::imwrite(image_path, noisy_collage);

            // Создание и сохранение метаданных
            json metadata = createMetadata(dist, snr_db);
            std::string json_file_name = "d" + std::to_string(dist) + "_snr" + std::to_string((int)snr_db) + "dB.json";
            std::string json_path = "../src/gt/" + json_file_name;
            std::ofstream json_file(json_path);
            json_file << metadata.dump(4);
        }
    }
}

void ImageGenerator::parseConfig(const json &config)
{
    distribution = config.value("distribution", 0);
    snr_db = config.value("snr_db", 50.0);
    seed = config.value("seed", 0);
}

void ImageGenerator::generate_default_config(const std::string &path)
{
    json j;
    j["distribution"] = 0;
    j["snr_db"] = 50.0;
    j["seed"] = 0;

    std::ofstream o(path);
    o << std::setw(4) << j << std::endl;
}