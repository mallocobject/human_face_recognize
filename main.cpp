#include "image_reader.h"
#include "image_shower.hpp"
// #include "pca_with_evd.h"
#include "pca_with_svd.h"

void PCAUseSVD(int k)
{
    ImageReader image_reader("/home/liudan/human_face_recognize/YaleB", 3, 30, 11);
    PCAWithSVD pca_svd;
    pca_svd.decompose(image_reader.getTrainSet());
    pca_svd.setTruncEigenNum(k);
    SHOW(pca_svd.centerVector());
    for (int i = 0; i < k; i++)
    {
        SHOW(pca_svd.U().col(i), std::to_string(pca_svd.eigenValueVector()(i)));
    }
    pca_svd.setEncodedTrain(pca_svd.encodeAll(image_reader.getTrainSet()));
    int ret = pca_svd.calc(pca_svd.encodeAll(image_reader.getTestSet()), image_reader.getLabels(),
                           image_reader.samples_per_person());
    std::cout << "result:" << ret << '/' << image_reader.getTestSize() << std::endl;
    SHOW(image_reader.at(0, false));
    SHOW(pca_svd.reconstruct(pca_svd.encode(image_reader.at(0, false))));
}

int main(int argc, char *argv[])
{
    if (argc > 1)
    {
        PCAUseSVD(std::atoi(argv[1]));
    }
    else
    {
        PCAUseSVD(17);
    }
}