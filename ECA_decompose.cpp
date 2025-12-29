#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <eigen3/Eigen/src/SVD/JacobiSVD.h>
#include <iostream>
#include <sys/types.h>

int main()
{
    // (features, samples)
    Eigen::MatrixXf X(3, 2);
    X << 1, 2, 3, 4, 5, 6;

    std::cout << "matrix X:\n" << X << std::endl;

    Eigen::VectorXf mean = X.rowwise().mean();
    Eigen::MatrixXf X_centered = X.colwise() - mean;

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(X_centered, Eigen::ComputeThinU);

    Eigen::MatrixXf U = svd.matrixU();
    Eigen::MatrixXf sigma = svd.singularValues().asDiagonal();

    std::cout << "U:\n" << U << std::endl;
    std::cout << "Σ:\n" << sigma << std::endl;

    Eigen::MatrixXf X_projected = U.transpose() * X_centered;
    std::cout << "after projection:\n" << X_projected << std::endl;
}
