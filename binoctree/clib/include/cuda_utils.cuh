#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H


#include <stdio.h>
#include <stdlib.h>
#include <utility>
#include <math.h>
#include <cmath>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>


#define TOTAL_THREADS 512
#define N_BLOCKS(batch_size) ((batch_size + TOTAL_THREADS - 1) / TOTAL_THREADS)

// inline int opt_n_threads(int work_size) {
//   const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

//   return max(min(1 << pow_2, TOTAL_THREADS), 1);
// }

// inline dim3 opt_block_config(int x, int y) {
//   const int x_threads = opt_n_threads(x);
//   const int y_threads =
//       max(min(opt_n_threads(y), TOTAL_THREADS / x_threads), 1);
//   dim3 block_config(x_threads, y_threads, 1);

//   return block_config;
// }

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


#define CUDA_CHECK_ERRORS()                                           \
  do {                                                                \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
      fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",  \
              cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__, \
              __FILE__);                                              \
      exit(-1);                                                       \
    }                                                                 \
  } while (0)

#endif
