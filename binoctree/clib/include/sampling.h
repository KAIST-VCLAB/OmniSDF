#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> uniform_sampling(
    torch::Tensor min_depth,
    torch::Tensor max_depth,
    torch::Tensor sample_dist,
    torch::Tensor near,
    torch::Tensor far,
    torch::Tensor hits,
    const int n_sample,
    const float sample_boundary
);