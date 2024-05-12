#include "cuda_utils.cuh"
#include "cuda_math.cuh"


__global__ void uniform_sampling_kernel(
    int n_rays, int n_voxels, int n_samples, float sample_boundary,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> min_depth,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> max_depth,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> sample_dist,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> near,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> far,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> hits,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> z_vals,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> z_dist
) {
    const int ray_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (ray_idx >= n_rays) {return;}
    float MAX_DEPTH = 10000.0 - 1;

    if (hits[ray_idx] > 0.5) {
        float bound_min = min_depth[ray_idx][0] - sample_dist[ray_idx] * sample_boundary;
        float bound_max = max_depth[ray_idx][0] + sample_dist[ray_idx] * sample_boundary;

        int sample_cnt = 0;
        int step = 0;
        float z = bound_min;

        while(true) {
            if (sample_cnt == n_samples) break;
            if (z > MAX_DEPTH) break;
            
            z_vals[ray_idx][sample_cnt] = z;
            
            z = z + sample_dist[ray_idx];
            sample_cnt++;

            if (z > bound_max) {
                // Step to next intersecting voxel's bound
                step++;
                bound_min = min_depth[ray_idx][step] - sample_dist[ray_idx] * sample_boundary;
                bound_max = max_depth[ray_idx][step] + sample_dist[ray_idx] * sample_boundary;
                
                // Update sample distance
                if (z < bound_min) {
                    z = bound_min;
                }
            }
        }
        z_dist[ray_idx] = sample_dist[ray_idx];
    }
    else
    {
        float sample_step = (far[ray_idx] - near[ray_idx]) / n_samples;
        float z = near[ray_idx];
        for (int k=0; k < n_samples; k++) {
            z_vals[ray_idx][k] = z;
            z = z + sample_step;
        }
        z_dist[ray_idx] = sample_step;
    }
}



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
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    dim3 BLOCK_DIM(TOTAL_THREADS, 1, 1);
    dim3 GRID_DIM(N_BLOCKS(n_rays), 1, 1);

    uniform_sampling_kernel<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
        n_rays, n_voxels, n_sample, sample_boundary,
        min_depth.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        max_depth.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        sample_dist.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        near.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        far.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        hits.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        z_vals.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        z_dist.packed_accessor32<float, 1, torch::RestrictPtrTraits>()
    );

    CUDA_CHECK_ERRORS();
}
