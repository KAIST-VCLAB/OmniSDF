#include "cuda_utils.cuh"
#include "intersection.h"


void tree_cuboid_intersection_kernel_wrapper(
    int n_rays, int n_voxels, int max_intersection,
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor node_vertices,
    torch::Tensor node_children,
    torch::Tensor node_sample,
    torch::Tensor intersection_idx,
    torch::Tensor min_depths,
    torch::Tensor max_depths
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tree_cuboid_intersection(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor node_vertices,
    torch::Tensor node_children,
    torch::Tensor node_sample,
    int max_intersection
) {
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(node_vertices);
    
    torch::Tensor intersect_idx = torch::zeros(
        {rays_o.size(0), max_intersection},
        torch::device(rays_o.device()).dtype(torch::ScalarType::Int).requires_grad(false));

    torch::Tensor min_depths = torch::zeros(
        {rays_o.size(0), max_intersection},
        torch::device(rays_o.device()).dtype(torch::ScalarType::Float).requires_grad(false));
    
    torch::Tensor max_depths = torch::zeros(
        {rays_o.size(0), max_intersection},
        torch::device(rays_o.device()).dtype(torch::ScalarType::Float).requires_grad(false));

    tree_cuboid_intersection_kernel_wrapper(
        rays_o.size(0),
        node_vertices.size(0),
        max_intersection,
        rays_o, rays_d, 
        node_vertices,
        node_children,
        node_sample,
        intersect_idx, 
        min_depths,
        max_depths
    );

    return std::make_tuple(intersect_idx, min_depths, max_depths);
}