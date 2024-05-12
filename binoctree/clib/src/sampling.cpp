#include "cuda_utils.cuh"
#include "intersection.h"


void uniform_sampling_kernel_wrapper(
    int n_rays, int n_voxels, int n_sample,
    float sample_boundary,
    torch::Tensor min_depth,
    torch::Tensor max_depth,
    torch::Tensor sample_dist,
    torch::Tensor near,
    torch::Tensor far,
    torch::Tensor hits,
    torch::Tensor z_vals,
    torch::Tensor z_dist

);

std::tuple<torch::Tensor, torch::Tensor> uniform_sampling(
    torch::Tensor min_depth,
    torch::Tensor max_depth,
    torch::Tensor sample_dist,
    torch::Tensor near,
    torch::Tensor far,
    torch::Tensor hits,
    const int n_sample,
    const float sample_boundary
) {
    CHECK_INPUT(min_depth);
    CHECK_INPUT(max_depth);
    CHECK_INPUT(sample_dist);
        
    torch::Tensor z_vals = torch::zeros(
        {min_depth.size(0), n_sample},
        torch::device(min_depth.device()).dtype(torch::ScalarType::Float).requires_grad(false));
    
    torch::Tensor z_dist = torch::zeros(
        {min_depth.size(0)},
        torch::device(min_depth.device()).dtype(torch::ScalarType::Float).requires_grad(false));

    uniform_sampling_kernel_wrapper(
        min_depth.size(0),
        min_depth.size(1),
        n_sample,
        sample_boundary,
        min_depth, max_depth, sample_dist, near, far, hits, z_vals, z_dist
    );

    return std::make_tuple(z_vals, z_dist);
}