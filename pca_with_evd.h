#ifndef PCA_WITH_EVD_H
#define PCA_WITH_EVD_H

#include "pca.hpp"
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <eigen3/Eigen/src/Eigenvalues/EigenSolver.h>

class PCAWithEVD : public PCA<PCAWithEVD>
{
  public:
    void decomposeImpl(const Eigen::MatrixXf &matrix)
    {
        center_vector_ = matrix.rowwise().mean();
        assert(center_vector_.size() == matrix.rows());

        Eigen::MatrixXf centered_mat = matrix.colwise() - center_vector_;
        Eigen::MatrixXf cov_mat = centered_mat * centered_mat.transpose() / (centered_mat.rows() - 1);
        Eigen::EigenSolver<Eigen::MatrixXf> evd(cov_mat);
        eigen_value_vector_ = evd.eigenvalues().real();
        U_ = evd.eigenvectors().real();
        assert(U_.rows() == matrix.rows());
    }

    Eigen::VectorXf reconstructImpl(const Eigen::VectorXf &encoded)
    {
        int k = tunc_eigen_num_;
        Eigen::VectorXf reconstructed =
            U_.leftCols(k) * eigen_value_vector_.head(k).asDiagonal() * encoded + center_vector_;
        return reconstructed;
    }
};

#endif