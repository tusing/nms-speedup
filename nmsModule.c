#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <unistd.h>
#include <CL/cl.h>

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

void reportOCLError(cl_int err, char* string);
static char string[100];
void CHK_ERR(cl_int err, int line);

void CHK_ERR(cl_int err, int line) {
    if(err != CL_SUCCESS) {
        reportOCLError(err, string);
        printf("Error: %s, File: %s, Line: %d\n", string, __FILE__, line);
        exit(-1);
    }
}

void reportOCLError(cl_int err, char* string) {
    switch (err) {
        case CL_DEVICE_NOT_FOUND:
            strcpy(string, "Device not found.");
            break;
        case CL_DEVICE_NOT_AVAILABLE:
            strcpy(string, "Device not available");
            break;
        case CL_COMPILER_NOT_AVAILABLE:     
            strcpy(string, "Compiler not available");
            break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            strcpy(string, "Memory object allocation failure");
            break;
        case CL_OUT_OF_RESOURCES:
            strcpy(string, "Out of resources");
            break;
        case CL_OUT_OF_HOST_MEMORY:
            strcpy(string, "Out of host memory");
            break;
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            strcpy(string, "Profiling information not available");
            break;
        case CL_MEM_COPY_OVERLAP:
            strcpy(string, "Memory copy overlap");
            break;
        case CL_IMAGE_FORMAT_MISMATCH:
            strcpy(string, "Image format mismatch");
            break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            strcpy(string, "Image format not supported");
            break;
        case CL_BUILD_PROGRAM_FAILURE:     
            strcpy(string, "Program build failure");
            break;
        case CL_MAP_FAILURE:         
            strcpy(string, "Map failure");
            break;
        case CL_INVALID_VALUE:
            strcpy(string, "Invalid value");
            break;
        case CL_INVALID_DEVICE_TYPE:
            strcpy(string, "Invalid device type");
            break;
        case CL_INVALID_PLATFORM:        
            strcpy(string, "Invalid platform");
            break;
        case CL_INVALID_DEVICE:     
            strcpy(string, "Invalid device");
            break;
        case CL_INVALID_CONTEXT:        
            strcpy(string, "Invalid context");
            break;
        case CL_INVALID_QUEUE_PROPERTIES: 
            strcpy(string, "Invalid queue properties");
            break;
        case CL_INVALID_COMMAND_QUEUE:          
            strcpy(string, "Invalid command queue");
            break;
        case CL_INVALID_HOST_PTR:            
            strcpy(string, "Invalid host pointer");
            break;
        case CL_INVALID_MEM_OBJECT:              
            strcpy(string, "Invalid memory object");
            break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  
            strcpy(string, "Invalid image format descriptor");
            break;
        case CL_INVALID_IMAGE_SIZE:           
            strcpy(string, "Invalid image size");
            break;
        case CL_INVALID_SAMPLER:     
            strcpy(string, "Invalid sampler");
            break;
        case CL_INVALID_BINARY:                    
            strcpy(string, "Invalid binary");
            break;
        case CL_INVALID_BUILD_OPTIONS:           
            strcpy(string, "Invalid build options");
            break;
        case CL_INVALID_PROGRAM:               
            strcpy(string, "Invalid program");
            break;
        case CL_INVALID_PROGRAM_EXECUTABLE:  
            strcpy(string, "Invalid program executable");
            break;
        case CL_INVALID_KERNEL_NAME:         
            strcpy(string, "Invalid kernel name");
            break;
        case CL_INVALID_KERNEL_DEFINITION:      
            strcpy(string, "Invalid kernel definition");
            break;
        case CL_INVALID_KERNEL:               
            strcpy(string, "Invalid kernel");
            break;
        case CL_INVALID_ARG_INDEX:           
            strcpy(string, "Invalid argument index");
            break;
        case CL_INVALID_ARG_VALUE:               
            strcpy(string, "Invalid argument value");
            break;
        case CL_INVALID_ARG_SIZE:              
            strcpy(string, "Invalid argument size");
            break;
        case CL_INVALID_KERNEL_ARGS:           
            strcpy(string, "Invalid kernel arguments");
            break;
        case CL_INVALID_WORK_DIMENSION:       
            strcpy(string, "Invalid work dimension");
            break;
        case CL_INVALID_WORK_GROUP_SIZE:          
            strcpy(string, "Invalid work group size");
            break;
        case CL_INVALID_WORK_ITEM_SIZE:      
            strcpy(string, "Invalid work item size");
            break;
        case CL_INVALID_GLOBAL_OFFSET: 
            strcpy(string, "Invalid global offset");
            break;
        case CL_INVALID_EVENT_WAIT_LIST: 
            strcpy(string, "Invalid event wait list");
            break;
        case CL_INVALID_EVENT:                
            strcpy(string, "Invalid event");
            break;
        case CL_INVALID_OPERATION:       
            strcpy(string, "Invalid operation");
            break;
        case CL_INVALID_GL_OBJECT:              
            strcpy(string, "Invalid OpenGL object");
            break;
        case CL_INVALID_BUFFER_SIZE:          
            strcpy(string, "Invalid buffer size");
            break;
        case CL_INVALID_MIP_LEVEL:
            strcpy(string, "Invalid mip-map level");   
            break;
        default:
            strcpy(string, "Unknown");
            break;
    }
 }

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
void nms_c_src(float *xmins, float *ymins, float* widths, float* heights, int *order, int *keep, float threshold, int n, float *probs) {
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
void nms_omp_src(float *xmins, float *ymins, float* widths, float* heights, int *order, int *keep, float threshold, int n, float *probs) {
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

/* parallel C implementation, without probability sorting */
void nms_c_unsorted_src(float *xmins, float *ymins, float* widths, float* heights, int *order, int *keep, float threshold, int n, float *probs) {
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
            if(iou_result > threshold) {
                if (probs[i] > probs[j]) {
                    keep[j] = 0;
                } else {
                    keep[i] = 0;
                    break;
                }
            }
        }
    }
}

/* Vectorized implementation of NMS, for benchmarking */
void nms_simd_src(float *xmins, float *ymins, float* widths, float* heights, int *order, int *keep, float threshold, int n, float *probs) {
    for(int i=0; i<n; i++) {
        if(keep[i] == 0) {
            continue;
        }
        #pragma omp parallel for firstprivate(xmins, ymins, widths, heights, order, keep, threshold, n, i)
        for (int j=i+1; j <= n - 8; j += 8) {
            __m256 results = simd_lowerleft_iou(xmins, ymins, widths, heights, i, j);
            float* iouresult = (float*)&results;
            for (int k = 0; k < 8; k++) {
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
            j++;
        }
    }
}

/* GPU implementation of NMS, for benchmarking, Partially sourced from previous homeworks. */
void nms_gpu_src(float *xmins, float *ymins, float *widths, float *heights, int *order, int *keep, float threshold, int n, float *probs) {
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
    cl_mem g_xmins, g_ymins, g_widths, g_heights, g_order, g_keep; //, g_probs;
    g_xmins = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*n, NULL, &ret);
    CHK_ERR(ret, __LINE__);
    g_ymins = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*n, NULL, &ret);
    CHK_ERR(ret, __LINE__);
    g_widths = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*n, NULL, &ret);
    CHK_ERR(ret, __LINE__);
    g_heights = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*n, NULL, &ret);
    CHK_ERR(ret, __LINE__);
    g_order = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*n, NULL, &ret);
    CHK_ERR(ret, __LINE__);
    g_keep = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*n, NULL, &ret);
    CHK_ERR(ret, __LINE__);

    /* Copy data from host CPU to GPU */
    ret = clEnqueueWriteBuffer(command_queue, g_xmins, 1, 0, sizeof(float)*n, xmins, 0, NULL, NULL);
    CHK_ERR(ret, __LINE__);
    ret = clEnqueueWriteBuffer(command_queue, g_ymins, 1, 0, sizeof(float)*n, ymins, 0, NULL, NULL);
    CHK_ERR(ret, __LINE__);
    ret = clEnqueueWriteBuffer(command_queue, g_widths, 1, 0, sizeof(float)*n, widths, 0, NULL, NULL);
    CHK_ERR(ret, __LINE__);
    ret = clEnqueueWriteBuffer(command_queue, g_heights, 1, 0, sizeof(float)*n, heights, 0, NULL, NULL);
    CHK_ERR(ret, __LINE__);
    ret = clEnqueueWriteBuffer(command_queue, g_order, 1, 0, sizeof(int)*n, order, 0, NULL, NULL);
    CHK_ERR(ret, __LINE__);
    ret = clEnqueueWriteBuffer(command_queue, g_keep, 1, 0, sizeof(int)*n, keep, 0, NULL, NULL);
    CHK_ERR(ret, __LINE__);

    /* Set OpenCL Kernel Arguments */
    ret = clSetKernelArg(nms, 0, sizeof(cl_mem), &g_xmins);
    CHK_ERR(ret, __LINE__);
    ret = clSetKernelArg(nms, 1, sizeof(cl_mem), &g_ymins);
    CHK_ERR(ret, __LINE__);
    ret = clSetKernelArg(nms, 2, sizeof(cl_mem), &g_widths);
    CHK_ERR(ret, __LINE__);
    ret = clSetKernelArg(nms, 3, sizeof(cl_mem), &g_heights);
    CHK_ERR(ret, __LINE__);
    ret = clSetKernelArg(nms, 4, sizeof(cl_mem), &g_order);
    CHK_ERR(ret, __LINE__);
    ret = clSetKernelArg(nms, 5, sizeof(cl_mem), &g_keep);
    CHK_ERR(ret, __LINE__);
    ret = clSetKernelArg(nms, 6, sizeof(float), &threshold);
    CHK_ERR(ret, __LINE__);
    ret = clSetKernelArg(nms, 7, sizeof(int), &n);
    CHK_ERR(ret, __LINE__);

    /* Define the global and local workgroup sizes */
    size_t global_work_size[1] = {n*(n-1) + (128 - (n*(n-1) % 128))};
    size_t local_work_size[1] = {128};

    /* Call kernel on the GPU */
    ret = clEnqueueNDRangeKernel(command_queue,
                                 nms,
                                 1,                 //work_dim,
                                 NULL,              //global_work_offset
                                 global_work_size,  //global_work_size
                                 local_work_size,   //local_work_size
                                 0,                 //num_events_in_wait_list
                                 NULL,              //event_wait_list
                                 NULL);
    CHK_ERR(ret, __LINE__);

    /* Read result of GPU on host CPU */
    ret = clEnqueueReadBuffer(command_queue, g_keep, 1, 0, sizeof(float)*n, keep, 0, NULL, NULL);
    CHK_ERR(ret, __LINE__);

    /* Shut down the OpenCL runtime */
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(nms);
    ret = clReleaseProgram(program);

    clReleaseMemObject(g_xmins);
    clReleaseMemObject(g_ymins);
    clReleaseMemObject(g_widths);
    clReleaseMemObject(g_heights);
    clReleaseMemObject(g_order);
    clReleaseMemObject(g_keep);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
}
