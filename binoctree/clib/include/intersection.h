#include <torch/extension.h>
#include <tuple>

torch::Tensor sphere_intersection(
    torch::Tensor rays_o, torch::Tensor rays_d, float radius
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cuboid_intersection(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor node_vertices,
    int max_intersection
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tree_cuboid_intersection(
    torch::Tensor rays_o,
    torch::Tensor rays_d,
    torch::Tensor node_vertices,
    torch::Tensor node_children,
    torch::Tensor node_sample,
    int max_intersection
);
