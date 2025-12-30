#include "image_reader.h"
#include "image_shower.hpp"
#include "pca.hpp"

int main(int argc, char *argv[])
{
    ImageReader image_reader("./Yale", 15, 10, 1);
    ImageShower image_shower;
    image_shower.show(image_reader.at(0, true));
}