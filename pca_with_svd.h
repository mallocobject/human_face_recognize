#ifndef PCA_WITH_SVD_H
#define PCA_WITH_SVD_H

#include "pca.hpp"
#include <cassert>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <eigen3/Eigen/src/SVD/JacobiSVD.h>
#include <iostream>

class PCAWithSVD : public PCA<PCAWithSVD>
{
  private:
    Eigen::VectorXf singular_value_vector_;

  public:
    void decomposeImpl(const Eigen::MatrixXf &matrix)
    {
        center_vector_ = matrix.rowwise().mean();
        assert(center_vector_.size() == matrix.rows());

        Eigen::MatrixXf centered_mat = matrix.colwise() - center_vector_;

        Eigen::JacobiSVD<Eigen::MatrixXf> svd(centered_mat, Eigen::ComputeThinU);
        singular_value_vector_ = svd.singularValues();
        eigen_value_vector_ = singular_value_vector_.array().square() / (matrix.cols() - 1);
        U_ = svd.matrixU();
        assert(U_.rows() == matrix.rows());
    }

    Eigen::VectorXf reconstructImpl(const Eigen::VectorXf &encoded)
    {
        int k = tunc_eigen_num_;
        Eigen::VectorXf reconstructed =
            U_.leftCols(k) * singular_value_vector_.head(k).asDiagonal() * encoded + center_vector_;
        return reconstructed;
    }
};

#endif