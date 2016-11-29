#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

float lowerleft_iou(float* xmins, float* ymins, float* widths, float* heights, int i, int j) {
    float x0 = max(xmins[i], xmins[j]);
    float y0 = max(ymins[i], ymins[j]);
    float x1 = min(xmins[i] + widths[i], xmins[j] + widths[j]);
    float y1 = min(ymins[i] + heights[i], ymins[j] + heights[j]);


    float box1Area = widths[i] * heights[i];
    float box2Area = widths[j] * heights[j];

    float union_area = 0;
    if (x1 > x0) {
        union_area = (x1 - x0) * (y1 - y0);
    }

    union_area = max(union_area, 0);
    float tot_area = box1Area + box2Area - union_area;
    float retval = 0;

    if (tot_area > 0.0) {
        retval = min(1.0, union_area / tot_area); /* Round to 1 bc fp division is sketchy */
    }
    return retval;
}

__kernel void nms(__global float *xmins, __global float *ymins, __global float *widths, __global float *heights, __global int *order, __global int *keep, float threshold, int n) {

    // size_t tid = get_local_id(0);	// where am I in the workgroup?
    // size_t gid = get_group_id(0);	// which workgroup am I in?
    // size_t dim = get_local_size(0);	// how large is my workgroup?
    size_t idx = get_global_id(0);	// where am I in the global index?

    int i = (idx / n); //floor division
    int j = i + (idx % n) + 1;
    if (j < n) {
        float iou_result = 0;
        float x0 = max(xmins[i], xmins[j]);
        float y0 = max(ymins[i], ymins[j]);
        float x1 = min(xmins[i] + widths[i], xmins[j] + widths[j]);
        float y1 = min(ymins[i] + heights[i], ymins[j] + heights[j]);


        float box1Area = widths[i] * heights[i];
        float box2Area = widths[j] * heights[j];

        float union_area = 0;
        if (x1 > x0) {
            union_area = (x1 - x0) * (y1 - y0);
        }

        union_area = max(union_area, 0);
        float tot_area = box1Area + box2Area - union_area;
        float retval = 0;

        if (tot_area > 0.0) {
            iou_result = min(1.0, union_area / tot_area); /* Round to 1 bc fp division is sketchy */
        }
        // float iou_result = lowerleft_iou(xmins, ymins, widths, heights, idx, j);
        // float iou_result = 0;
        if (iou_result > threshold) {
            keep[j] = 0;
        }
    }
}

// __kernel void alt_nms(__global float *xmins, __global float *ymins, __global float *widths, __global float *heights, __global int *order, __global int *keep, float threshold, int n) {

//     size_t tid = get_local_id(0);   // where am I in the workgroup?
//     size_t gid = get_group_id(0);   // which workgroup am I in?
//     size_t dim = get_local_size(0); // how large is my workgroup?
//     size_t idx = get_global_id(0);  // where am I in the global index?

//     i = (idx / n) //floor division
//     j = i + (idx % n) + 1
//     if (j < n) {
//         float iou_result = lowerleft_iou(xmins, ymins, widths, heights, i, j);
//         if (iou_result > threshold) {
//             keep[j] = 0;
//         }
//     }
// }

