#ifndef IMAGE_READER_H
#define IMAGE_READER_H

#include <cassert>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <string>

class ImageReader
{

  private:
    std::string base_path_;
    int num_persons_;
    int num_train_per_person_;
    int num_test_per_person_;
    Eigen::MatrixXf train_dataset_;
    Eigen::MatrixXf test_dataset_;
    Eigen::VectorXi labels_;

  public:
    ImageReader(const char *base_path, int num_persons, int num_train_per_person, int num_test_per_person);
    void readTrainSet();
    void readTestSet();
    void generateLabel();

    Eigen::VectorXf at(int index, bool is_train)
    {
        return is_train ? train_dataset_.col(index) : test_dataset_.col(index);
    }

    Eigen::MatrixXf getTrainSet()
    {
        return train_dataset_;
    }

    Eigen::MatrixXf getTestSet()
    {
        return test_dataset_;
    }
};

#endif