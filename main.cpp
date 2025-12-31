#include "image_reader.h"
#include "image_shower.hpp"
#include "pca_with_evd.h"
#include "pca_with_svd.h"
#include <string>

void PCAUseSVD()
{
    ImageReader image_reader("/home/liudan/human_face_recognize/Yale", 15, 10, 1);
    PCAWithSVD pca_svd;
    pca_svd.decompose(image_reader.getTrainSet());
    pca_svd.setTruncEigenNum(15);
    SHOW(pca_svd.centerVector());
    for (int i = 0; i < 15; i++)
    {
        SHOW(pca_svd.U().col(i), std::to_string(pca_svd.eigenValueVector()(i)));
    }
    pca_svd.setEncodedTrain(pca_svd.encodeAll(image_reader.getTrainSet()));
    int ret = pca_svd.calc(pca_svd.encodeAll(image_reader.getTestSet()));
    std::cout << "result:" << ret << '/' << image_reader.category() << std::endl;
    SHOW(image_reader.at(0, false));
    SHOW(pca_svd.reconstruct(pca_svd.encode(image_reader.at(0, false))));
}

// very very ... very slowly! I have no patience on waiting it complete.
void PCAUseEVD()
{
    ImageReader image_reader("/home/liudan/human_face_recognize/Yale", 15, 10, 1);
    PCAWithEVD pca_evd;
    pca_evd.decompose(image_reader.getTrainSet());
    pca_evd.setTruncEigenNum(15);
    SHOW(pca_evd.centerVector());
    pca_evd.setEncodedTrain(pca_evd.encodeAll(image_reader.getTrainSet()));
    int ret = pca_evd.calc(pca_evd.encodeAll(image_reader.getTestSet()));
    std::cout << "result:" << ret << '/' << image_reader.category() << std::endl;
    SHOW(image_reader.at(0, false));
    SHOW(pca_evd.reconstruct(pca_evd.encode(image_reader.at(0, false))));
}

int main(int argc, char *argv[])
{
    PCAUseSVD();
}