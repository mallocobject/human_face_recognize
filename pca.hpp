#ifndef PCA_H
#define PCA_H

#include <cassert>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <iostream>

template <typename VD> class PCA
{
  protected:
    Eigen::VectorXf center_vector_;
    Eigen::MatrixXf encoded_vectors_;
    Eigen::MatrixXf U_; // the subset of eigen vectors(Thin)
    // Eigen::MatrixXf V_; // the position int face space
    Eigen::VectorXf eigen_value_vector_;
    int tunc_eigen_num_;

  public:
    Eigen::VectorXf centerVector()
    {
        return center_vector_;
    }

    Eigen::VectorXf eigenValueVector()
    {
        return eigen_value_vector_;
    }

    Eigen::MatrixXf eigenValueDiagonal()
    {
        return eigen_value_vector_.asDiagonal();
    }

    Eigen::MatrixXf U()
    {
        return U_;
    }

    void decompose(const Eigen::MatrixXf &matrix)
    {
        // center_vector_.resize(matrix.cols());
        // encoded_vectors_.resize(matrix.rows(), matrix.cols());

        static_cast<VD *>(this)->decomposeImpl(matrix);
    }

    void setTruncEigenNum(int k)
    {
        tunc_eigen_num_ = k;
        assert(k > 0 && k <= eigen_value_vector_.size());
    }

    Eigen::VectorXf encode(const Eigen::VectorXf &vector)
    {
        int k = tunc_eigen_num_;
        Eigen::MatrixXf projector = U_.leftCols(k).transpose();
        return projector * (vector - center_vector_);
    }

    Eigen::VectorXf reconstruct(const Eigen::VectorXf &encoded)
    {
        int k = tunc_eigen_num_;
        return static_cast<VD *>(this)->reconstructImpl(encoded);
    }

    void setEncodedTrain(const Eigen::MatrixXf &matrix)
    {
        encoded_vectors_ = matrix;
    }

    Eigen::MatrixXf encodeAll(const Eigen::MatrixXf &matrix)
    {
        int k = tunc_eigen_num_;
        Eigen::MatrixXf projector = U_.leftCols(k).transpose();

        return projector * (matrix.colwise() - center_vector_);
    }

    // return the number of corrent recognization
    int calc(const Eigen::MatrixXf &matrix, const Eigen::VectorXi &labels, int samples_per_person)
    {
        // std::cout << matrix.cols() << std::endl;
        int corrent_cnt = 0;

        for (int i = 0; i < matrix.cols(); i++)
        {
            Eigen::VectorXf test = matrix.col(i);

            double min_distance = std::numeric_limits<double>::max();
            int predicted_label = -1;
            int min_index = -1;

            double distance = 0;
            for (int j = 0; j < encoded_vectors_.cols(); j++)
            {
                distance = (test - encoded_vectors_.col(j)).norm();
                if (distance < min_distance)
                {
                    min_distance = distance;
                    predicted_label = j / samples_per_person;
                }

                // distance += (test - encoded_vectors_.col(j)).norm();

                // if ((j + 1) % samples_per_person == 0)
                // {
                //     distance /= samples_per_person;

                //     if (distance < min_distance)
                //     {
                //         min_distance = distance;
                //         predicted_label = j / samples_per_person;
                //     }

                //     distance = 0;
                // }
            }

            // std::cout << predicted_label + 1 << ':' << labels(i) << std::endl;
            if (predicted_label + 1 == labels(i))
            {
                corrent_cnt++;
            }
        }
        return corrent_cnt;
    }
};

#endif