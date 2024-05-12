#include "cuda_utils.cuh"
#include "cuda_math.cuh"
#include <stack>


__global__ void tree_cuboid_intersection_kernel(
    int n_rays, int n_voxels, int max_intersection,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_o,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_d,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> node_vertices,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> node_children,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> node_sample,
    torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> intersection_idx,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> min_depth,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> max_depth
) {
    const int ray_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (ray_idx >= n_rays) {return;}
    int cnt = 0;

    for (int i=0; i < max_intersection; i++) {
        intersection_idx[ray_idx][i] = -1;
        min_depth[ray_idx][i] = -1;
        max_depth[ray_idx][i] = -1;
    }

    float3 ray_o = make_float3(rays_o[ray_idx][0], rays_o[ray_idx][1], rays_o[ray_idx][2]);
    float3 ray_d = make_float3(rays_d[ray_idx][0], rays_d[ray_idx][1], rays_d[ray_idx][2]);
    
    bool flag= false;

    int rear=1, v=-1;
    int stack[256] = {-1}; // Actually queue
    stack[0] = 0; // root
    while(rear > 0 && rear < 256 && cnt < max_intersection) {
        // pop
        v = stack[0];
        for (int i=1; i<rear; i++) {
            stack[i-1] = stack[i];
        }
        rear--;

        if (v == 0){
            flag = true;
        } else {
            flag = false;
            // Intersection
            float3 v0 = make_float3(node_vertices[v][0][0], node_vertices[v][0][1], node_vertices[v][0][2]);
            float3 v1 = make_float3(node_vertices[v][1][0], node_vertices[v][1][1], node_vertices[v][1][2]);
            float3 v2 = make_float3(node_vertices[v][2][0], node_vertices[v][2][1], node_vertices[v][2][2]);
            float3 v3 = make_float3(node_vertices[v][3][0], node_vertices[v][3][1], node_vertices[v][3][2]);
            float3 v4 = make_float3(node_vertices[v][4][0], node_vertices[v][4][1], node_vertices[v][4][2]);
            float3 v5 = make_float3(node_vertices[v][5][0], node_vertices[v][5][1], node_vertices[v][5][2]);
            float3 v6 = make_float3(node_vertices[v][6][0], node_vertices[v][6][1], node_vertices[v][6][2]);
            float3 v7 = make_float3(node_vertices[v][7][0], node_vertices[v][7][1], node_vertices[v][7][2]);

            float2 depths =  cube_intersection(
                ray_idx, ray_o, ray_d,
                v0, v1, v2, v3, v4, v5, v6, v7
            );

            if(depths.x > 0 and depths.y > 0) {
                flag = true;
                if (node_sample[v]) {
                    flag = false;
                    intersection_idx[ray_idx][cnt] = v;
                    min_depth[ray_idx][cnt] = depths.x;
                    max_depth[ray_idx][cnt] = depths.y;
                    cnt++;
                    if (cnt == max_intersection) {
                        break;
                    }
                }
            }
        }
        // Intersection
        if (flag) {
            int n_child = node_children[v][8];
            for (int i=0; i<n_child; i++) {
                stack[rear] = node_children[v][i];
                rear++;
            }
        }
    }
}

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
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    dim3 BLOCK_DIM(TOTAL_THREADS, 1, 1);
    dim3 GRID_DIM(N_BLOCKS(n_rays), 1, 1);

    tree_cuboid_intersection_kernel<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
        n_rays, n_voxels, max_intersection,
        rays_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        rays_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        node_vertices.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        node_children.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        node_sample.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        intersection_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        min_depths.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        max_depths.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    CUDA_CHECK_ERRORS();
}
