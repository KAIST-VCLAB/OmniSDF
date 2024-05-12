#include "cuda_utils.cuh"
#include "intersection.h"

void sphere_intersection_cuda(
    int n_data, float radius, torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor z_vals
);

torch::Tensor sphere_intersection(torch::Tensor rays_o, torch::Tensor rays_d, float radius) {
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);

    int n_data = rays_o.size(0);
    auto tensor_opt = torch::device(rays_o.device()).dtype(at::ScalarType::Float).requires_grad(false);
    
    torch::Tensor z_vals = torch::zeros({n_data}, tensor_opt);
    sphere_intersection_cuda(n_data, radius, rays_o, rays_d, z_vals);
    return z_vals;
}