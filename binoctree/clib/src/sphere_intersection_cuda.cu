#include "cuda_utils.cuh"
#include "cuda_math.cuh"

__global__ void sphere_intersection_cuda_kernel(
    int n_rays, float radius,
    float* __restrict__ rays_o,
    float* __restrict__ rays_d,
    float* __restrict__ z_vals
) {
    const int ray_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (ray_idx >= n_rays) {return;}
    float3 ray_o = make_float3(rays_o[ray_idx*3], rays_o[ray_idx*3+1], rays_o[ray_idx*3+2]);
    float3 ray_d = make_float3(rays_d[ray_idx*3], rays_d[ray_idx*3+1], rays_d[ray_idx*3+2]);

    float qa = dot3f(ray_d, ray_d);
    float qb = dot3f(ray_d, ray_o);
    float qc = dot3f(ray_o, ray_o) - radius*radius;
    float det = qb * qb - qa * qc;
    if (det >= 0)
    {
        float t1 = (-qb + sqrt(det)) / qa;
        float t2 = (-qb - sqrt(det)) / qa;

        float t = -1; // t1 > t2
        if (t1 > 0) { t = t1; }
        if (t2 > 0) { t = t2; }
        if (t > 0) {
            z_vals[ray_idx] = t;
        }
    }
}

void sphere_intersection_cuda(
    int n_data, float radius, torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor z_vals
) {
    dim3 BLOCK_DIM(TOTAL_THREADS, 1, 1);
    dim3 GRID_DIM(N_BLOCKS(n_data), 1 , 1);

    sphere_intersection_cuda_kernel<<<GRID_DIM, BLOCK_DIM, 0>>>(
        n_data, radius, rays_o.data_ptr<float>(), rays_d.data_ptr<float>(), z_vals.data_ptr<float>()
    );
    cudaDeviceSynchronize();
}