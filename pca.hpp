#ifndef PCA_H
#define PCA_H

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>

template <typename VD> class PCA
{
  protected:
    Eigen::VectorXf center_vector_;
    Eigen::MatrixXf encoded_vectors_;
    Eigen::MatrixXf face_space_projector_; // the subset of eigen vectors
    Eigen::VectorXf eigen_values_;

  public:
    void decompose(const Eigen::MatrixXf &matrix)
    {
        center_vector_.resize(matrix.cols());
        encoded_vectors_.resize(matrix.rows(), matrix.cols());

        static_cast<VD *>(this)->decompose();
    }

    Eigen::VectorXf encode(const Eigen::VectorXf &vector)
    {
        return face_space_projector_ * vector;
    }

    // return the number of corrent recognization
    int calc(const Eigen::MatrixXf &matrix)
    {
        int corrent_cnt = 0;

        for (int i = 0; i < matrix.cols(); i++)
        {
            Eigen::VectorXf test = matrix.col(i);

            float min_distance = std::numeric_limits<float>::max();
            int predicted_label = -1;
            int min_index = -1;

            for (int j = 0; j < encoded_vectors_.cols(); j++)
            {
                float distance = (test - encoded_vectors_.col(j)).norm();
                if (distance < min_distance)
                {
                    min_distance = distance;
                    predicted_label = j / 10;
                    min_index = j;
                }
            }

            if (predicted_label == i)
            {
                corrent_cnt++;
            }
        }
        return corrent_cnt;
    }
};

#endif