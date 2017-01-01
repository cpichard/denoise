
#include <ImfArray.h>
//#include <ImfChannelList.h>
//#include <ImfInputFile.h>
//#include <ImfOutputFile.h>
#include <ImfRgba.h>
//#include <ImfRgbaFile.h>

bool readEXR(const std::string &fileName, Imf::Array2D<Imf::Rgba> &pixels,
             unsigned int &width, unsigned int &height);
bool writeExr(const std::string &fileName, const Imf::Array2D<Imf::Rgba> &pixels, 
        unsigned int &width, unsigned int &height);
