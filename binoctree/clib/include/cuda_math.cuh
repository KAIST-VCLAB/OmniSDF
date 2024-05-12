#include "cuda_runtime.h"
#include <math.h>

#define PI  3.14159265358979323846
#define EPS std::numeric_limits<float>::epsilon()

__device__ __inline__ float3 convert_c2s( const float3 &cpts ) {
    float r, theta, phi = 0.0;
    r = sqrt(cpts.x*cpts.x + cpts.y*cpts.y + cpts.z*cpts.z);
    theta = acos(cpts.z / r);
    if (cpts.x != 0) {
        phi = atan2(cpts.y, cpts.x);
    }
    if (phi < 0) {
        phi = phi + 2 * PI;
    }
    return make_float3(r, theta, phi);
}
__device__ __inline__ float3 convert_s2c( const float3 &spts ) {
    float r = spts.x;
    float theta = spts.y;
    float phi = spts.z;
    float x = r * sin(theta) * cos(phi);
    float y = r * sin(theta) * sin(phi);
    float z = r * cos(theta);
    return make_float3(x, y, z);
}
__device__ __inline__ float dot3f(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __inline__ float3 cross3f(const float3 &a, const float3 &b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}
__device__ __inline__ float3 emul3f(const float3 &a, const float3 &b) {
    return make_float3( a.x*b.x, a.y*b.y, a.z*b.z );
}
__device__ __inline__ float3 add3f(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __inline__ float3 subtract3f(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __inline__ float magnitude3f(const float3 &a) {
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}
__device__ __inline__ float3 norm3f(const float3 &a) {
    float norm = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    return make_float3(a.x/norm, a.y/norm, a.z/norm);
}
__device__ __inline__ float distance3f(const float3 &a, const float3 &b) {
    float3 d =  subtract3f(a, b);
    return sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
}
__device__ __inline__ float3 ray_march3f(const float3 &o, const float3 &d, float t) {
    return make_float3(
        o.x + d.x * t,
        o.y + d.y * t,
        o.z + d.z * t);
}
__device__ __inline__ bool locate_points(
    const float3 &xyz, const float3 &voxel_start, const float3 &voxel_end
) {
    if (
        voxel_start.x <= xyz.x && xyz.x < voxel_end.x && 
        voxel_start.y <= xyz.y && xyz.y < voxel_end.y &&
        voxel_start.z <= xyz.z && xyz.z < voxel_end.z
    ) {
        return true;
    }

    return false;
}

__device__ __inline__ float triangle_intersection(
    const int ray_idx,
    const float3 &ray_o, const float3 &ray_d,
    const float3 &v0, const float3 &v1, const float3 &v2
) {
    float3 n = norm3f(cross3f(subtract3f(v1, v0), subtract3f(v2, v0)));
    float D = -dot3f(n, v0);
    float lmda_n = (dot3f(n, ray_o) + D);
    float lmda_d = dot3f(n, ray_d);

    if (lmda_d > 0) {
        float t = -lmda_n/lmda_d;
        if (t < 0) { return 0; }
        float3 p = ray_march3f(ray_o, ray_d, t);
        float3 edge0 = subtract3f(v1, v0);
        float3 edge1 = subtract3f(v2, v1);
        float3 edge2 = subtract3f(v0, v2);
        float3 c0 = subtract3f(p, v0);
        float3 c1 = subtract3f(p, v1);
        float3 c2 = subtract3f(p, v2);
        if (dot3f(n, cross3f(edge0, c0)) >= 0 &&
            dot3f(n, cross3f(edge1, c1)) >= 0 &&
            dot3f(n, cross3f(edge2, c2)) >= 0
        ) { return t; } // CCW
        // if (dot3f(n, cross3f(edge0, c0)) >= - EPS &&
        //     dot3f(n, cross3f(edge1, c1)) >= - EPS &&
        //     dot3f(n, cross3f(edge2, c2)) >= - EPS
        // ) { return t; } // CCW
    }
    return 0;
}


__device__ __inline__ float2 cube_intersection(
    const int ray_idx,
    const float3 &ray_o, const float3 &ray_d,
    const float3 &v0, const float3 &v1, const float3 &v2, const float3 &v3,
    const float3 &v4, const float3 &v5, const float3 &v6, const float3 &v7
) {
    float t_max[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float t_min[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    t_max[0] = triangle_intersection(ray_idx, ray_o, ray_d, v0, v1, v2);
    t_max[1] = triangle_intersection(ray_idx, ray_o, ray_d, v1, v3, v2);
    t_max[2] = triangle_intersection(ray_idx, ray_o, ray_d, v0, v2, v4);
    t_max[3] = triangle_intersection(ray_idx, ray_o, ray_d, v2, v6, v4);
    t_max[4] = triangle_intersection(ray_idx, ray_o, ray_d, v0, v4, v1);
    t_max[5] = triangle_intersection(ray_idx, ray_o, ray_d, v4, v5, v1);
    t_max[6] = triangle_intersection(ray_idx, ray_o, ray_d, v1, v5, v3);
    t_max[7] = triangle_intersection(ray_idx, ray_o, ray_d, v5, v7, v3);
    t_max[8] = triangle_intersection(ray_idx, ray_o, ray_d, v2, v3, v7);
    t_max[9] = triangle_intersection(ray_idx, ray_o, ray_d, v2, v7, v6);
    t_max[10] = triangle_intersection(ray_idx, ray_o, ray_d, v4, v6, v7);
    t_max[11] = triangle_intersection(ray_idx, ray_o, ray_d, v7, v5, v4);

    t_min[0] = triangle_intersection(ray_idx, ray_o, ray_d, v2, v1, v0);
    t_min[1] = triangle_intersection(ray_idx, ray_o, ray_d, v2, v3, v1);
    t_min[2] = triangle_intersection(ray_idx, ray_o, ray_d, v4, v2, v0);
    t_min[3] = triangle_intersection(ray_idx, ray_o, ray_d, v4, v6, v2);
    t_min[4] = triangle_intersection(ray_idx, ray_o, ray_d, v1, v4, v0);
    t_min[5] = triangle_intersection(ray_idx, ray_o, ray_d, v1, v5, v4);
    t_min[6] = triangle_intersection(ray_idx, ray_o, ray_d, v3, v5, v1);
    t_min[7] = triangle_intersection(ray_idx, ray_o, ray_d, v3, v7, v5);
    t_min[8] = triangle_intersection(ray_idx, ray_o, ray_d, v7, v3, v2);
    t_min[9] = triangle_intersection(ray_idx, ray_o, ray_d, v6, v7, v2);
    t_min[10] = triangle_intersection(ray_idx, ray_o, ray_d, v7, v6, v4);
    t_min[11] = triangle_intersection(ray_idx, ray_o, ray_d, v4, v5, v7);
    
    float t_near=0;
    float t_far=0;
    int face1 = 0;
    int face2 = 0;
    for (int i=0; i<12; i++){
        if (t_min[i] > 0) {
            t_near = t_min[i];
            face1 = i;
        }
        if (t_max[i] > 0) {
            t_far = t_max[i];
            face2 = i;
        }
    }    
    
    return make_float2(t_near, t_far);
}