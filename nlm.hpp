/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef IMAGE_DENOISING_H
#define IMAGE_DENOISING_H

typedef float4 TColor;
#include <driver_types.h>

////////////////////////////////////////////////////////////////////////////////
// Filter configuration
////////////////////////////////////////////////////////////////////////////////
#define NLM_WINDOW_RADIUS   7
#define NLM_BLOCK_RADIUS    5
#define NLM_WINDOW_AREA     ( (2 * NLM_WINDOW_RADIUS + 1) * (2 * NLM_WINDOW_RADIUS + 1) )
#define INV_NLM_WINDOW_AREA ( 1.0f / (float)NLM_WINDOW_AREA )

#define NLM_WEIGHT_THRESHOLD    0.1f
#define NLM_LERP_THRESHOLD      0.1f

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8

#ifndef MAX
#define MAX(a,b) ((a < b) ? b : a)
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif

// CUDA wrapper functions for allocation/freeing texture arrays
//extern "C" cudaError_t CUDA_Bind2TextureArray();
//extern "C" cudaError_t CUDA_UnbindTexture();
//extern "C" cudaError_t CUDA_MallocArray(uchar4 **h_Src, int imageW, int imageH);
//extern "C" cudaError_t CUDA_FreeArray();

// CUDA kernel functions
extern "C" void cuda_Copy(TColor *d_dst, int imageW, int imageH);
extern "C" void cuda_NLM(TColor *d_dst, int imageW, int imageH, float Noise, float lerpC);
extern "C" void cuda_NLMdiag(TColor *d_dst, int imageW, int imageH, float Noise, float lerpC);
extern "C" void bindTexture(cudaArray* cu_array, cudaChannelFormatDesc &channelDesc);
extern "C" void unbindTexture();
#endif
