// #include <filesystem> // C++17 文件系统
// #include <iostream>
// #include <opencv2/opencv.hpp>

// namespace fs = std::filesystem;

// void processImage(const std::string &imagePath)
// {
//     try
//     {
//         // 尝试读取图像，转换为灰度图
//         cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

//         // 如果图像为空，抛出异常
//         if (image.empty())
//         {
//             throw cv::Exception(cv::Error::StsError, "Failed to load image", "cv::imread", __FILE__, __LINE__);
//         }

//         // 输出图像的宽度和高度
//         std::cout << "Image: " << imagePath << " | Size: " << image.cols << "x" << image.rows << std::endl;

//         // 创建窗口并调整大小
//         cv::namedWindow("Image", cv::WINDOW_NORMAL);
//         cv::resizeWindow("Image", 800, 600);

//         // 显示图像
//         cv::imshow("Image", image);
//         cv::waitKey(0);
//         cv::destroyWindow("Image");
//     }
//     catch (const cv::Exception &e)
//     {
//         std::cerr << "OpenCV Error for " << imagePath << ": " << e.what() << std::endl;
//     }
// }

// int main(int argc, char *argv[])
// {
//     if (argc < 2)
//     {
//         std::cerr << "Error: No image path or directory provided!" << std::endl;
//         return -1;
//     }

//     std::string path = argv[1];

//     // 检查路径是否是文件夹
//     if (fs::is_directory(path))
//     {
//         std::cout << "Directory provided. Processing all images in the folder..." << std::endl;

//         // 遍历文件夹中的所有文件
//         for (const auto &entry : fs::directory_iterator(path))
//         {
//             if (entry.is_regular_file())
//             {
//                 std::string filePath = entry.path().string();
//                 processImage(filePath); // 处理每一个图像文件
//             }
//         }
//     }
//     else
//     {
//         // 如果提供的是单个文件路径，则直接处理
//         processImage(path);
//     }

//     return 0;
// }
