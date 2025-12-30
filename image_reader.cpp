#include "image_reader.h"
#include "common.h"
#include <cassert>
#include <sstream>

ImageReader::ImageReader(const char *base_path, int num_persons, int num_train_per_person, int num_test_per_person)
    : base_path_(base_path), num_persons_(num_persons), num_train_per_person_(num_train_per_person),
      num_test_per_person_(num_test_per_person)
{
    if (base_path_.back() != '/')
    {
        base_path_.push_back('/');
    }

    int total_pixels = ROW_PIXELS * COL_PIXELS;
    int total_train = num_persons_ * num_train_per_person_;
    int total_test = num_persons_ * num_test_per_person_;

    train_dataset_.resize(total_pixels, total_train);
    test_dataset_.resize(total_pixels, total_test);
    labels_.resize(num_persons_);

    generateLabel();
    readTrainSet();
    readTestSet();
}

void ImageReader::generateLabel()
{
    for (int i = 0; i < num_persons_; i++)
    {
        labels_(i) = i + 1;
    }
}

void ImageReader::readTrainSet()
{
    int total_pixels = ROW_PIXELS * COL_PIXELS;

    for (int i = 0; i < num_persons_; i++)
    {
        for (int j = 0; j < num_train_per_person_; j++)
        {
            std::ostringstream filename;
            filename << base_path_ << i + 1 << "/s" << j + 1 << ".bmp";
            cv::Mat img = cv::imread(filename.str(), cv::IMREAD_GRAYSCALE);
            assert(img.rows == ROW_PIXELS && img.cols == COL_PIXELS);

            Eigen::Matrix<uchar, ROW_PIXELS, COL_PIXELS, Eigen::RowMajor> eigen_mat;
            cv::cv2eigen(img, eigen_mat);

            Eigen::Matrix<float, ROW_PIXELS, COL_PIXELS, Eigen::RowMajor> float_img = eigen_mat.cast<float>();

            train_dataset_.col(i * num_train_per_person_ + j) = float_img.reshaped<Eigen::ColMajor>(total_pixels, 1);
        }
    }
}

void ImageReader::readTestSet()
{
    int total_pixels = ROW_PIXELS * COL_PIXELS;

    for (int i = 0; i < num_persons_; i++)
    {

        std::ostringstream filename;
        filename << base_path_ << i + 1 << "/s11.bmp";
        cv::Mat img = cv::imread(filename.str(), cv::IMREAD_GRAYSCALE);
        assert(img.rows == ROW_PIXELS && img.cols == COL_PIXELS);

        Eigen::Matrix<uchar, ROW_PIXELS, COL_PIXELS, Eigen::RowMajor> eigen_mat;
        cv::cv2eigen(img, eigen_mat);

        Eigen::Matrix<float, ROW_PIXELS, COL_PIXELS, Eigen::RowMajor> float_img = eigen_mat.cast<float>();

        test_dataset_.col(i) = float_img.reshaped<Eigen::ColMajor>(total_pixels, 1);
    }
}