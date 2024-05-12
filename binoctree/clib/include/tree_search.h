#include <torch/extension.h>

torch::Tensor search_node(
    torch::Tensor points, torch::Tensor voxel_specs, torch::Tensor childrens, const int max_level
);

torch::Tensor search_leaf_node(
    torch::Tensor points, torch::Tensor voxel_specs, torch::Tensor childrens, const int max_level
);
