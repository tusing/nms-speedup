#include <string.h>
#include <stdio.h>
#include <stdlib.h>
// #include <string>
#include <math.h>
#include <unistd.h>
#include <CL/cl.h>

/* GPU implementation of NMS, for benchmarking, Partially sourced from previous homeworks. */
void nms_gpu_src(float *h_boxes, int *h_order, int *h_keep, float h_threshold, int h_n) {
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
    cl_mem g_boxes, g_order, g_keep;
    g_boxes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, &ret);
    //CHK_ERR(ret);
    g_order = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * n, NULL, &ret);
    //CHK_ERR(ret);
    g_keep = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * n, NULL, &ret);
    //CHK_ERR(ret);

    /* Copy data from host CPU to GPU */
    ret = clEnqueueWriteBuffer(command_queue, g_boxes, true, 0, sizeof(float) * n, h_boxes, 0, NULL, NULL);
    //CHK_ERR(ret);
    ret = clEnqueueWriteBuffer(command_queue, g_order, true, 0, sizeof(int) * n, h_order, 0, NULL, NULL);
    //CHK_ERR(ret);
    ret = clEnqueueWriteBuffer(command_queue, g_keep, true, 0, sizeof(int) * n, h_keep, 0, NULL, NULL);
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
                                 1,                 //work_dim,
                                 NULL,              //global_work_offset
                                 global_work_size,  //global_work_size
                                 local_work_size,   //local_work_size
                                 0,                 //num_events_in_wait_list
                                 NULL,              //event_wait_list
                                 NULL);
    //CHK_ERR(ret);

    /* Read result of GPU on host CPU */
    ret = clEnqueueReadBuffer(command_queue, g_keep, true, 0, sizeof(float) * n, h_keep, 0, NULL, NULL);
    //CHK_ERR(ret);

    for (int i = 0; i < n; i++) {
        printf("value=%d", h_keep[i]);
    }


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

int main(int argc, char *argv[]) {
    int n = (1 << 12);
    float *h_boxes = new float[n];
    int *h_order = new int[n];
    int *h_keep = new int[n];

    for (int i = 0; i < n; i++) {
        h_boxes[i] = 0.0;
        h_order[i] = 0;
        h_keep[i] = 0;
    }

    nms_gpu_src(h_boxes, h_order, h_keep, 2.0, 5);
}

