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

__m128 simd_lowerleft_iou(float* restrict xmins, float* restrict ymins, float* widths, float* heights, int i, int j) {
    // determine the (x, y)-coordinates of the intersection rectangle
    __m128 xminsi = _mm_broadcast_ss(xmins+i);
    __m128 xminsj = _mm_loadu_ps(xmins+j);
    __m128 yminsi = _mm_broadcast_ss(ymins+i);
    __m128 yminsj = _mm_loadu_ps(ymins+j);

    __m128 widthsi = _mm_broadcast_ss(widths+i);
    __m128 widthsj = _mm_loadu_ps(widths+j);
    __m128 heightsi = _mm_broadcast_ss(heights+i);
    __m128 heightsj = _mm_loadu_ps(heights+j);

    __m128 x0 = _mm_max_ps(xminsi, xminsj);
    __m128 y0 = _mm_max_ps(yminsi, yminsj);
    __m128 x1 = _mm_min_ps(_mm_add_ps(xminsi, widthsi), _mm_add_ps(xminsj, widthsj));
    __m128 y1 = _mm_min_ps(_mm_add_ps(yminsi, heightsi), _mm_add_ps(yminsj, heightsj));

    __m128 box1Area = _mm_mul_ps(widthsi, heightsi);
    __m128 box2Area = _mm_mul_ps(widthsj, heightsj);

    __m128 t0 = _mm_setzero_ps();
    __m128 t1 = _mm_mul_ps(_mm_sub_ps(x1, x0), _mm_sub_ps(y1, y0));
    __m128 mask = _mm_cmp_ps(x1, x0, _CMP_GT_OS);

    __m128 union_area = _mm_blendv_ps(t0, t1 , mask);
    union_area = _mm_max_ps(union_area, t0);

    __m128 tot_area = _mm_sub_ps(_mm_add_ps(box1Area, box2Area), union_area);

    float t = 1.0;
    __m128 t2 = _mm_broadcast_ss(&t);;
    __m128 t3 = _mm_min_ps(t2, _mm_div_ps(union_area, tot_area));
    __m128 retval = _mm_setzero_ps();
    mask = _mm_cmp_ps(tot_area, t0, 2);
    retval = _mm_blendv_ps(t3, retval, mask);
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

    __m128 t0 = _mm_broadcast_ps((void*) &threshold);
    int* fmask = (int*)malloc(4*sizeof(int));

    for(int i=0; i<n; i++) {
        if(keep[i] == 0) {
            continue;
        }

        #pragma omp parallel for
        for (int j=i+1; j <= n - 4; j += 4){
            {
                __m128 results = simd_lowerleft_iou(xmins, ymins, widths, heights, i, j);
                float* iouresult = (float*)&results;
                for (int k = 0; k < 4; k++) {
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
        int j = n - 4;
        while (j < n) {
            float iou_result = lowerleft_iou(xmins, ymins, widths, heights, i, j);
            if(iou_result > threshold) {
                keep[j] = 0;

            }
            j ++;
        }
    }
}
