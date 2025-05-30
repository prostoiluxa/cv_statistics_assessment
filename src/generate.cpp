#include <opencv2/opencv.hpp>
#include <iostream>
#include <generator.h>

int main() {
    ImageGenerator generator(42);
    generator.generateAll();
    return 0;
}

// int main(int argc, char** argv) {
//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0] << " <config_path> [image_path] [gt_path] [--seed seed_value]" << std::endl;
//         return 1;
//     }

//     std::string config_path = argv[1];
//     config_path = "../src/config/" + config_path;

//     if (argc == 2) {
//         ImageGenerator::generate_default_config(config_path);
//         std::cout << "Default config generated at " << config_path << std::endl;
//         return 0;
//     }

//     if (argc < 4) {
//         std::cerr << "Insufficient arguments. Expected image path and gt path." << std::endl;
//         return 1;
//     }

//     std::string image_path = argv[2];
//     image_path = "../src/test_images/" + image_path;
//     std::string gt_path = argv[3];
//     gt_path = "../src/gt/" + gt_path;

//     int seed = -1;
//     for (int i = 4; i < argc; ++i) {
//         if (std::string(argv[i]) == "--seed" && i + 1 < argc) {
//             seed = std::stoi(argv[i + 1]);
//             break;
//         }
//     }

//     ImageGenerator generator(config_path, seed);
//     cv::Mat image = generator.generate_collage(gt_path);

//     if (!cv::imwrite(image_path, image)) {
//         throw std::runtime_error("Failed to save image to: " + image_path);
//     }

//     std::cout << "Successfully generated:\n"
//               << "Image: " << image_path << "\n"
//               << "Ground truth: " << gt_path << std::endl;

//     return 0;

// }