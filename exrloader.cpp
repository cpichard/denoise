
// Loading an EXR file
#include "exrloader.hpp"
#include <ImathBox.h>
#include <ImfRgbaFile.h>

bool readEXR(const std::string &fileName, Imf::Array2D<Imf::Rgba> &pixels,
             unsigned int &width, unsigned int &height) {

    Imf::RgbaInputFile file(fileName.c_str());
    Imath::Box2i dataWindow = file.header().dataWindow();

    width = dataWindow.max.x - dataWindow.min.x + 1;
    height = dataWindow.max.y - dataWindow.min.y + 1;

    // Make sure that we can handle empty images correctly
    if (width * height < 1) {
        return false;
    }

    pixels.resizeErase(height, width);
    file.setFrameBuffer(&pixels[0][0] - dataWindow.min.x - dataWindow.min.y*width, 1, width);
    file.readPixels(dataWindow.min.y, dataWindow.max.y);

    return true;
}

bool writeExr(const std::string &fileName, const Imf::Array2D<Imf::Rgba> &pixels, 
        unsigned int &width, unsigned int &height)
{
    // Write output
    Imf::RgbaOutputFile fileDst (fileName.c_str(), width, height, Imf::WRITE_RGBA);
    fileDst.setFrameBuffer (&pixels[0][0], 1, width);
    fileDst.writePixels (height);
    return true;
}
