#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#include <unistd.h>
#include <CL/cl.h>

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

void nms_c_unsorted_src(float *xmins, float *ymins, float* widths, float* heights, int *order, int *keep, float threshold, int n, int *probs) {
    #pragma omp parallel for
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
                if (probs[i] > probs[j]) {
                    keep[j] = 0;
                } else {
                    keep[i] = 0;
                    break;
                }
                //printf("%f\n", iou_result);
                // keep[j] = 0;
            }
        }
    }


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
void nms_c_src(float *xmins, float *ymins, float* widths, float* heights, int *order, int *keep, float threshold, int n, int *probs) {
    for(int i=0; i<n; i++) {
        if(keep[i] == 0) {
            continue;
        }
        for(int j=i+1; j<n; j++) {
            float iou_result = lowerleft_iou(xmins, ymins, widths, heights, i, j);
            if(iou_result > threshold) {
                keep[j] = 0;
            }
        }
    }


}

/* OpenMP Implementation */
void nms_omp_src(float *xmins, float *ymins, float* widths, float* heights, int *order, int *keep, float threshold, int n, int *probs) {

    for(int i=0; i<n; i++) {
        if(keep[i] == 0) {
            continue;
        }
        #pragma omp parallel for schedule(dynamic, 8) firstprivate(xmins, ymins, widths, heights, order, keep, threshold, n, i)
        for(int j=i+1; j<n; j++) {
            if (keep[j] == 0) {
                continue;
            }
            float iou_result = lowerleft_iou(xmins, ymins, widths, heights, i, j);
            if(iou_result > threshold) {
                keep[j] = 0;
            }
        }
    }

}
/* OpenMP Implementation 2*/
void nms_omp1_src(float *xmins, float* ymins, float* widths, float* heights, int *order, int *keep, float threshold, int n) {
#pragma omp parallel for schedule(dynamic, 8) firstprivate(xmins, ymins, widths, heights, order, keep, threshold, n)
    for(int i=n-1; i>0; i-=1) {
        for(int j=0; j<i; j++) {
            float iou_result = lowerleft_iou(xmins, ymins, widths, heights, i, j);
            if(iou_result > threshold) {
                keep[i] = 0;
                continue;
            }
        }
    }

}

/* Vectorized implementation of NMS, for benchmarking */
void nms_simd_src(float *xmins, float *ymins, float* widths, float* heights, int *order, int *keep, float threshold, int n, int *probs) {

/* #pragma omp parallel for schedule(dynamic, 1) firstprivate(xmins, ymins, widths, heights, order, keep, threshold, n, probs) */
    for(int i=0; i<n; i++) {
        if(keep[i] == 0) {
            continue;
        }

#pragma omp parallel for firstprivate(xmins, ymins, widths, heights, order, keep, threshold, n, i)
        for (int j=i+1; j <= n - 8; j += 8){

            __m256 results = simd_lowerleft_iou(xmins, ymins, widths, heights, i, j);
            float* iouresult = (float*)&results;
            for (int k = 0; k < 8; k++) {
                //printf("%f\t%d\t%d\t%d\n", iouresult[k], i, j+k, order[j+k]);
                if (iouresult[k] > threshold) {
                    keep[j+k] = 0;
                }
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
void nms_gpu_src(float *h_xmins, int *h_ymins, int *h_widths, float h_heights, int *h_order, int *h_keep, float h_threshold, int h_n, int *h_probs) {

    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_program program = NULL;
    cl_kernel nms = NULL;
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    FILE *fp;
    char fileName[] = "./nms.cl";
    char *source_str;
    size_t source_size;

    int n = (1 << 12);

    /* Load the source code containing the kernel */
    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*) malloc(0x100000);
    source_size = fread(source_str, 1, 0x100000, fp);
    fclose(fp);

    /* Get Platform and Device Info */
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    /* Create OpenCL context */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    /* Create Command Queue */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    /* Create Kernel Program from the source */
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    /* Build Kernel Program */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    /* Create OpenCL kernel */
    nms = clCreateKernel(program, "nms", &ret);


    /* Create Memory Buffer */
    cl_mem g_xmins, g_ymins, g_keep;
    g_boxes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, &ret);
    //CHK_ERR(ret);
    g_order = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * n, NULL, &ret);
    //CHK_ERR(ret);
    g_keep = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * n, NULL, &ret);
    //CHK_ERR(ret);

    /* Copy data from host CPU to GPU */
    ret = clEnqueueWriteBuffer(command_queue, g_boxes, 1, 0, sizeof(float) * n,
                               h_boxes, 0, NULL, NULL);
    //CHK_ERR(ret);
    ret = clEnqueueWriteBuffer(command_queue, g_order, 1, 0, sizeof(int) * n,
                               h_order, 0, NULL, NULL);
    //CHK_ERR(ret);
    ret = clEnqueueWriteBuffer(command_queue, g_keep, 1, 0, sizeof(int) * n,
                               h_keep, 0, NULL, NULL);
    //CHK_ERR(ret);

    /* Set OpenCL Kernel Arguments */
    ret = clSetKernelArg(nms, 0, sizeof(cl_mem), &g_boxes);
    //CHK_ERR(ret);
    ret = clSetKernelArg(nms, 1, sizeof(cl_mem), &g_order);
    //CHK_ERR(ret);
    ret = clSetKernelArg(nms, 2, sizeof(cl_mem), &g_keep);
    //CHK_ERR(ret);
    ret = clSetKernelArg(nms, 3, sizeof(float), &h_threshold);
    //CHK_ERR(ret);
    ret = clSetKernelArg(nms, 4, sizeof(int), &h_n);
    //CHK_ERR(ret);

    /* Define the global and local workgroup sizes */
    size_t global_work_size[1] = {n};
    size_t local_work_size[1] = {128};


    /* Call kernel on the GPU */
    ret = clEnqueueNDRangeKernel(command_queue,
                                 nms,
                                 1,//work_dim,
                                 NULL, //global_work_offset
                                 global_work_size, //global_work_size
                                 local_work_size, //local_work_size
                                 0, //num_events_in_wait_list
                                 NULL, //event_wait_list
                                 NULL //
                                );
    //CHK_ERR(ret);

    /* Read result of GPU on host CPU */
    ret = clEnqueueReadBuffer(command_queue, g_keep, 1, 0, sizeof(float) * n,
                              h_keep, 0, NULL, NULL);
    //CHK_ERR(ret);

    for (int i = 0; i < n; i++)
        printf("value=%d", h_keep[i]);


    /* Shut down the OpenCL runtime */
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(nms);
    ret = clReleaseProgram(program);

    free(h_boxes);
    free(h_order);
    free(h_keep);

    clReleaseMemObject(g_boxes);
    clReleaseMemObject(g_order);
    clReleaseMemObject(g_keep);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
}
