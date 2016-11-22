__kernel void nms(__global float *boxes,
                  __global int *order,
                  __global int *keep,
                  float threshold,
                  int n) {
    size_t tid = get_local_id(0); // where am I in the workgroup?
    size_t gid = get_group_id(0); // which workgroup am I in?
    size_t dim = get_local_size(0); // how large is my workgroup?
    size_t idx = get_global_id(0); // where am I in the global index?

    for (int i = 0; i < dim; i++)
        keep[tid] += 1;
}

