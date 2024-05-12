#include "cuda_utils.cuh"
#include "tree_search.h"
#include "intersection.h"
#include "sampling.h"
#include <pybind11/pybind11.h>


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_sphere_intersection", &sphere_intersection);
    m.def("tree_cuboid_intersection", &tree_cuboid_intersection);
    m.def("uniform_sampling", &uniform_sampling);
};