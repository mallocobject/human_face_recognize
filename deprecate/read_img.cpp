#include <cassert>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <eigen3/Eigen/src/SVD/JacobiSVD.h>
#include <iostream>
#include <limits>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>

int main()
{
    std::string base_path = "Yale/";
    int num_persons = 15;
    int num_faces_per_person = 10;
    int num_imgs = num_persons * num_faces_per_person;

    Eigen::MatrixXd data_matrix(10000, num_imgs);

    for (int i = 0; i < num_persons; i++)
    {
        for (int j = 0; j < num_faces_per_person; j++)
        {
            std::ostringstream filename;
            filename << base_path << i + 1 << "/s" << j + 1 << ".bmp";

            cv::Mat img = cv::imread(filename.str(), cv::IMREAD_GRAYSCALE);
            Eigen::Matrix<uchar, 100, 100, Eigen::RowMajor> matrix;

            assert(img.rows == 100 && img.cols == 100);

            cv::cv2eigen(img, matrix);

            Eigen::Matrix<double, 100, 100, Eigen::RowMajor> float_img = matrix.cast<double>();

            data_matrix.col(i * num_faces_per_person + j) = float_img.reshaped<Eigen::ColMajor>(10000, 1);
        }
    }

    Eigen::VectorXd row_means = data_matrix.rowwise().mean();

    assert(row_means.size() == data_matrix.rows());

    Eigen::MatrixXd centered_matrix = data_matrix.colwise() - row_means;

    assert(centered_matrix.cols() == num_imgs);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered_matrix, Eigen::ComputeThinU | Eigen::ComputeFullV);

    Eigen::MatrixXd U = svd.matrixU();
    // Eigen::MatrixXd V = svd.matrixV();
    Eigen::VectorXd singular_values = svd.singularValues();

    // for (int i = 0; i < 150; i++)
    // {
    //     std::cout << singular_values(i) << " ";
    // }
    // std::cout << std::endl;

    int k = 15;
    assert(k > 0 && k <= num_imgs);

    Eigen::VectorXd top_singular_values = singular_values.head(k);
    Eigen::MatrixXd top_U = U.leftCols(k);
    // Eigen::MatrixXd top_V = V.leftCols(k);

    Eigen::MatrixXd encode_faces = top_U.transpose() * centered_matrix;

    Eigen::MatrixXd test_matrix(10000, num_persons);

    for (int i = 0; i < num_persons; i++)
    {
        std::ostringstream filename;
        filename << base_path << i + 1 << "/s11.bmp";
        cv::Mat img = cv::imread(filename.str(), cv::IMREAD_GRAYSCALE);
        Eigen::Matrix<uchar, 100, 100, Eigen::RowMajor> matrix;

        assert(img.rows == 100 && img.cols == 100);

        cv::cv2eigen(img, matrix);

        Eigen::Matrix<double, 100, 100, Eigen::RowMajor> float_img = matrix.cast<double>();

        test_matrix.col(i) = float_img.reshaped<Eigen::ColMajor>(10000, 1);
    }

    Eigen::MatrixXd centered_test_matrix = test_matrix.colwise() - row_means;

    Eigen::MatrixXd test_faces = top_U.transpose() * centered_test_matrix;

    int corrent_cnt = 0;

    for (int i = 0; i < num_persons; i++)
    {
        Eigen::VectorXd test_encoding = test_faces.col(i);

        double min_distance = std::numeric_limits<double>::max();
        int predicted_label = -1;
        int min_index = -1;

        for (int j = 0; j < num_imgs; j++)
        {
            double distance = (test_encoding - encode_faces.col(j)).norm();
            if (distance < min_distance)
            {
                min_distance = distance;
                predicted_label = j / 10 + 1;
                min_index = j;
            }
        }

        std::cout << "preduct: " << predicted_label << std::endl;
        std::cout << "similar index: " << min_index / 10 + 1 << ':' << min_index % 10 + 1 << std::endl;

        if (predicted_label == i + 1)
        {
            corrent_cnt++;
        }
    }

    std::cout << corrent_cnt << '/' << num_persons << std::endl;

    return 0;
}

// Eigen::MatrixXd Sigma_k = top_singular_values.asDiagonal();
// Eigen::MatrixXd reconstructed_centered = top_U * Sigma_k * top_V.transpose();
// Eigen::MatrixXd reconstructed_original = reconstructed_centered.colwise() + row_means;

// int img_idx = 0;
// assert(img_idx >= 0 && img_idx < num_imgs);

// Eigen::VectorXd recon_col = reconstructed_original.col(img_idx);

// // 使用 Eigen 的 reshape 转换为 100×100 矩阵，然后转换为 cv::Mat
// Eigen::MatrixXd recon_mat = recon_col.reshaped<Eigen::ColMajor>(100, 100);

// cv::Mat display_img;
// cv::eigen2cv(recon_mat, display_img);

// cv::Mat normalized_img;
// cv::normalize(display_img, normalized_img, 0, 255, cv::NORM_MINMAX, CV_8U);

// cv::namedWindow("Reconstructed Image", cv::WINDOW_NORMAL); // 创建可调大小的窗口
// cv::resizeWindow("Reconstructed Image", 800, 600);         // 调整窗口的初始大小

// cv::imshow("Reconstructed Image", normalized_img);
// cv::waitKey(0);

// cv::destroyWindow("Reconstructed Image");

// Eigen::VectorXd singular_values = svd.singularValues().array().square() / (num_imgs - 1);

// for (int i = 0; i < 14; i++)
// {
//     std::cout << singular_values(i) << " ";
// }
// std::cout << std::endl;

// Eigen::MatrixXd sigma = singular_values.asDiagonal();

// std::cout << "U:\n" << U << std::endl;
// std::cout << "Σ:\n" << sigma << std::endl;