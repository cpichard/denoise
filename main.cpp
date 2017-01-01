#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <string.h>
#include "nlm.hpp"
#include "exrloader.hpp"
#include "cudaerror.hpp"

using namespace std;
using namespace OPENEXR_IMF_NAMESPACE;

int main(int argc, char **argv) {
    // Test files
    const char *srcFileName = "test_image_src.exr";
    const char *dstFileName = "test_image_dst.exr";

    // Read exr file
    Imf::Array2D<Imf::Rgba> hostSourcePixels; 
    unsigned int width, height;
    readEXR(srcFileName, hostSourcePixels, width, height);
    cout <<  "file size " << width << "X" << height << endl;

    // Convert half to float
    const unsigned int bufferSize = 4*sizeof(float)*width*height;
    float * const hostSourceFloatBuffer = (float*)malloc(bufferSize);
    float *bufferIterator = hostSourceFloatBuffer;
    for(int y=0; y<height; ++y) {
        for(int x=0; x<width; ++x) {
            const Imf::Rgba &pixel = hostSourcePixels[y][x];
            *bufferIterator++=pixel.r;
            *bufferIterator++=pixel.g;
            *bufferIterator++=pixel.b;
            *bufferIterator++=pixel.a;
        }
    }

    // allocate array and copy image data
    // Copy image to texture which is used as the source image
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaArray* deviceSourceFloatArray;
    checkCudaErrors(cudaMallocArray(&deviceSourceFloatArray, &channelDesc, width, height));
    cudaMemcpyToArray(deviceSourceFloatArray, 0, 0, hostSourceFloatBuffer, bufferSize, cudaMemcpyHostToDevice);

    // Bind the array to the texture
    bindTexture(deviceSourceFloatArray, channelDesc);
    
    // Allocate destination image
    TColor *deviceDestBuffer;
    cudaMalloc((void**) &deviceDestBuffer, bufferSize);

    // Process image
    static float lerpC = 0.01f;
    static float nlmNoise = 0.02f;
    cuda_NLM(deviceDestBuffer, width, height, 1.0f / (nlmNoise * nlmNoise), lerpC);

    // Get data back, convert to half 
    float *hostDestBuffer = (float*)malloc(bufferSize);
    checkCudaErrors(cudaMemcpy(hostDestBuffer, deviceDestBuffer, bufferSize, cudaMemcpyDeviceToHost));

    // Create array of half pixels
    Imf::Array2D<Imf::Rgba> hostDestPixels(height, width); 
    bufferIterator = hostDestBuffer;
    for(int y=0; y<height; ++y) {
        for(int x=0; x<width; ++x) {
            hostDestPixels[y][x].r = *bufferIterator++;
            hostDestPixels[y][x].g = *bufferIterator++;
            hostDestPixels[y][x].b = *bufferIterator++;
            hostDestPixels[y][x].a = *bufferIterator++;
        }
    }

    // Save exr file
    writeExr(std::string(dstFileName), hostDestPixels, width, height);

    // Free ressources 
    unbindTexture();
    free(hostDestBuffer);
    cudaFree(deviceDestBuffer);
    cudaFreeArray(deviceSourceFloatArray);
    free(hostSourceFloatBuffer);
}
