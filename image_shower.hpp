#ifndef IMAGE_SHOWER_H
#define IMAGE_SHOWER_H

#include "common.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <string>

class ImageShower
{
  public:
    static ImageShower &instance()
    {
        static ImageShower image_shower;
        return image_shower;
    }

    void show(const Eigen::VectorXf &eigen_vec, const std::string &title = "Image")
    {
        Eigen::MatrixXf eigen_mat = eigen_vec.reshaped<Eigen::ColMajor>(ROW_PIXELS, COL_PIXELS);
        cv::Mat cv_mat, cv_display;
        cv::eigen2cv(eigen_mat, cv_mat);
        cv::normalize(cv_mat, cv_display, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::namedWindow(title, cv::WINDOW_NORMAL);
        cv::resizeWindow(title, 800, 600);

        cv::imshow(title, cv_display);
        cv::waitKey(0);

        cv::destroyWindow(title);
    }

  private:
    ImageShower() = default;
};

#define SHOW(...) ImageShower::instance().show(__VA_ARGS__)

#endif