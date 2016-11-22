#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))



float lowerleft_iou(float* restrict xmins, float* restrict ymins, float* widths, float* heights, int i, int j) {
    // determine the (x, y)-coordinates of the intersection rectangle
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

__m256 simd_lowerleft_iou(float* restrict xmins, float* restrict ymins, float* widths, float* heights, int i, int j) {
    // determine the (x, y)-coordinates of the intersection rectangle
    __m256 xminsi = _mm256_broadcast_ss(xmins+i);
    __m256 xminsj = _mm256_loadu_ps(xmins+j);
    __m256 yminsi = _mm256_broadcast_ss(ymins+i);
    __m256 yminsj = _mm256_loadu_ps(ymins+j);

    __m256 widthsi = _mm256_broadcast_ss(widths+i);
    __m256 widthsj = _mm256_loadu_ps(widths+j);
    __m256 heightsi = _mm256_broadcast_ss(heights+i);
    __m256 heightsj = _mm256_loadu_ps(heights+j);

    __m256 x0 = _mm256_max_ps(xminsi, xminsj);
    __m256 y0 = _mm256_max_ps(yminsi, yminsj);
    __m256 x1 = _mm256_min_ps(_mm256_add_ps(xminsi, widthsi), _mm256_add_ps(xminsj, widthsj));
    __m256 y1 = _mm256_min_ps(_mm256_add_ps(yminsi, heightsi), _mm256_add_ps(yminsj, heightsj));

    __m256 box1Area = _mm256_mul_ps(widthsi, heightsi);
    __m256 box2Area = _mm256_mul_ps(widthsj, heightsj);

    __m256 t0 = _mm256_setzero_ps();
    __m256 t1 = _mm256_mul_ps(_mm256_sub_ps(x1, x0), _mm256_sub_ps(y1, y0));
    __m256 mask = _mm256_cmp_ps(x1, x0, _CMP_GT_OS);

    __m256 union_area = _mm256_blendv_ps(t0, t1 , mask);
    union_area = _mm256_max_ps(union_area, t0);

    __m256 tot_area = _mm256_sub_ps(_mm256_add_ps(box1Area, box2Area), union_area);

    float t = 1.0;
    __m256 t2 = _mm256_broadcast_ss(&t);;
    __m256 t3 = _mm256_min_ps(t2, _mm256_div_ps(union_area, tot_area));
    __m256 retval = _mm256_setzero_ps();
    mask = _mm256_cmp_ps(tot_area, t0, 2);
    retval = _mm256_blendv_ps(t3, retval, mask);
    return retval;

}

/* 	Scalar naive implementation of NMS, for benchmarking
	for i in range(len(order)):
		if not keep[order[i]]:
		    continue
		for j in range(i+1, len(order)):
		    if iou(boxes[order[i]], boxes[order[j]]) > threshold:
		        keep[order[j]] = False
	return keep
*/
void nms_c_src(float *xmins, float *ymins, float* widths, float* heights, int *order, int *keep, float threshold, int n) {
    for(int i=0; i<n; i++) {
        if(keep[i] == 0) {
            continue;
        }
        for(int j=i+1; j<n; j++) {
            if (keep[j] == 0) {
                continue;
            }
            float iou_result = lowerleft_iou(xmins, ymins, widths, heights, i, j);
            //printf("%f\t%d\t%d\t%d\n", iou_result, i, j, order[j]);
            if(iou_result > threshold) {
                //printf("%f\n", iou_result);
                keep[j] = 0;
            }
        }
    }


}

/* OpenMP Implementation */
void nms_omp_src(float *xmins, float* ymins, float* widths, float* heights, int *order, int *keep, float threshold, int n) {

    {
        for(int i=0; i<n; i++) {
            if(keep[i] == 0) {
                continue;
            }


            #pragma omp parallel for
            for(int j=i+1; j<n; j++) {
                if (keep[j] == 0) {
                    continue;
                }

                {
                    float iou_result = lowerleft_iou(xmins, ymins, widths, heights, i, j);
                    if(iou_result > threshold) {
                        keep[j] = 0;
                    }
                }
            }
        }
    }


}

/* Vectorized implementation of NMS, for benchmarking */
void nms_simd_src(float *xmins, float* ymins, float* widths, float* heights, int *order, int *keep, float threshold, int n) {

    __m256 t0 = _mm256_broadcast_ps((void*) &threshold);
    int* fmask = (int*)malloc(8*sizeof(int));

    for(int i=0; i<n; i++) {
        if(keep[i] == 0) {
            continue;
        }

        #pragma omp parallel for
        for (int j=i+1; j <= n - 8; j += 8){
            {
                __m256 results = simd_lowerleft_iou(xmins, ymins, widths, heights, i, j);
                float* iouresult = (float*)&results;
                for (int k = 0; k < 8; k++) {
                    //printf("%f\t%d\t%d\t%d\n", iouresult[k], i, j+k, order[j+k]);
                    if (iouresult[k] > threshold) {
                        keep[j+k] = 0;
                    }
                }
                /* while (j <= n - 8 && */
                /*        keep[j] == 0) { */
                /*     j++; */
                /* } */

            }

        }
        int j = n - 8;
        while (j < n) {
            float iou_result = lowerleft_iou(xmins, ymins, widths, heights, i, j);
            if(iou_result > threshold) {
                keep[j] = 0;

            }
            j ++;
        }
    }



}

/* GPU implementation of NMS, for benchmarking, Partially sourced from previous homeworks. */
void nms_gpu_src(float *h_boxes, int *h_order, int *h_keep, float h_threshold, int h_n) {
    /* std::string nms_kernel_str; */

    /* /\* Provide names of the OpenCL kernels */
    /*  * and cl file that they're kept in *\/ */
    /* std::string nms_name_str = */
    /*     std::string("nms"); */
    /* std::string nms_kernel_file = */
    /*     std::string("nms.cl"); */

    /* cl_vars_t cv; */
    /* cl_kernel nms; */

    /* /\* Read OpenCL file into STL string *\/ */
    /* readFile(nms_kernel_file, */
    /*          nms_kernel_str); */

    /* /\* Initialize the OpenCL runtime. *\/ */
    /* initialize_ocl(cv); */

    /* /\* Compile all OpenCL kernels *\/ */
    /* compile_ocl_program(nms, cv, nms_kernel_str.c_str(), */
    /*                     nms_name_str.c_str()); */

    /* /\* Arrays on the host are prefixed with 'h_'. *\/ */
    /* /\* Arrays on the device (GPU) *\/ */
    /* cl_mem g_boxes, g_order, g_keep; */

    /* /\* Allocate memory for arrays on the GPU *\/ */
    /* cl_int err = CL_SUCCESS; */

    /* /\* Create buffers on the GPU. *\/ */
    /* g_boxes = clCreateBuffer(cv.context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, &err); */
    /* CHK_ERR(err); */
    /* g_order = clCreateBuffer(cv.context, CL_MEM_READ_WRITE, sizeof(int) * n, NULL, &err); */
    /* CHK_ERR(err); */
    /* g_keep = clCreateBuffer(cv.context, CL_MEM_READ_WRITE, sizeof(int) * n, NULL, &err); */
    /* CHK_ERR(err); */

    /* /\* Copy data from host CPU to GPU *\/ */
    /* err = clEnqueueWriteBuffer(cv.commands, g_boxes, true, 0, sizeof(float) * n, */
    /*                            h_boxes, 0, NULL, NULL); */
    /* CHK_ERR(err); */
    /* err = clEnqueueWriteBuffer(cv.commands, g_order, true, 0, sizeof(int) * n, */
    /*                            h_order, 0, NULL, NULL); */
    /* CHK_ERR(err); */
    /* err = clEnqueueWriteBuffer(cv.commands, g_keep, true, 0, sizeof(int) * n, */
    /*                            h_keep, 0, NULL, NULL); */
    /* CHK_ERR(err); */


    /* /\* Define the global and local workgroup sizes *\/ */
    /* size_t global_work_size[1] = {n}; */
    /* size_t local_work_size[1] = {128}; */

    /* /\* Set Kernel Arguments *\/ */
    /* err = clSetKernelArg(nms, 0, sizeof(cl_mem), &g_boxes); */
    /* CHK_ERR(err); */
    /* err = clSetKernelArg(nms, 1, sizeof(cl_mem), &g_order); */
    /* CHK_ERR(err); */
    /* err = clSetKernelArg(nms, 2, sizeof(cl_mem), &g_keep); */
    /* CHK_ERR(err); */
    /* err = clSetKernelArg(nms, 3, sizeof(float), &h_threshold); */
    /* CHK_ERR(err); */
    /* err = clSetKernelArg(nms, 4, sizeof(int), &h_n); */
    /* CHK_ERR(err); */

    /* /\* Call kernel on the GPU *\/ */
    /* err = clEnqueueNDRangeKernel(cv.commands, */
    /*                              nms, */
    /*                              1,//work_dim, */
    /*                              NULL, //global_work_offset */
    /*                              global_work_size, //global_work_size */
    /*                              local_work_size, //local_work_size */
    /*                              0, //num_events_in_wait_list */
    /*                              NULL, //event_wait_list */
    /*                              NULL // */
    /*                             ); */
    /* CHK_ERR(err); */

    /* /\* Read result of GPU on host CPU *\/ */
    /* err = clEnqueueReadBuffer(cv.commands, g_keep, true, 0, sizeof(float) * n, */
    /*                           h_keep, 0, NULL, NULL); */
    /* CHK_ERR(err); */

    /* /\* Shut down the OpenCL runtime *\/ */
    /* uninitialize_ocl(cv); */

    /* delete [] h_boxes; */
    /* delete [] h_order; */
    /* delete [] h_keep; */

    /* clReleaseMemObject(g_boxes); */
    /* clReleaseMemObject(g_order); */
    /* clReleaseMemObject(g_keep); */
    /* return 0; */
}
