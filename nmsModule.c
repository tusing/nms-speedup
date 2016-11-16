#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

/* BICHEN'S ORIGINAL IOU CODE (wrong? negative values when both boxes same) */
/* DONT use this one, lowerleft_iou performs fewer computations*/
float center_iou(float *box1, float *box2) {
    float lr = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - 
        max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2]) + 1.0;
    if(lr > 0) {
        float tb = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - 
            max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3]) + 1.0;
        if(tb > 0) {
            float box_inxct= tb*lr;
            float box_union = box1[2]*box1[3] + box2[2]*box2[3] - box_inxct; 
            return box_inxct / box_union;
        }
    }
    return 0;
}

float lowerleft_iou(float *box1, float *box2) {
    // determine the (x, y)-coordinates of the intersection rectangle
    float x0 = max(box1[0], box2[0]);
    float y0 = max(box1[1], box2[1]);
    float x1 = min(box1[0] + box1[2], box2[0] + box2[2]);
    float y1 = min(box1[1] + box1[3], box2[1] + box2[3]);
    

    float box1Area = box1[2] * box1[3];
    float box2Area = box2[2] * box2[3];

    float union_area = 0;
    if (x1 > x0) {
        union_area = (x1 - x0) * (y1 - y0);
    }

    union_area = max(union_area, 0);
    float tot_area = box1Area + box2Area - union_area;
    
    if (tot_area > 0) {
        return min(1.0, union_area / tot_area); /* Round to 1 bc fp division is sketchy */
    } else {
        return 0;
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
void nms_c_src(float *boxes, int *order, int *keep, float threshold, int n) {

    for(int i=0; i<n; i++) {
        if(keep[order[i]] == 0) {
            continue;
        }
        for(int j=i+1; j<n; j++) {
            float iou_result = lowerleft_iou(boxes+4*order[i], boxes+4*order[j]);
            if(iou_result > threshold) {
                keep[order[j]] = 0;
            }   
        }
    }

    
}

/* Vectorized implementation of NMS, for benchmarking */
void nms_simd_src(float *boxes, int *order, int *keep, float threshold, int n) {
    for(int i=0; i<n; i++) {
        if(!keep[order[i]]) {
            continue;
        }
        //float boi = boxes + 4*order[i];
        for(int j=i+1; j<n; j+=4) {
            /* SIMD SOMETHING HERE 
               _m128 floats
               1. vectorize boxes
               2. constant vector filled with threshold
               3. vectorize iou on boxes
               4. simd compare iou and threshold
               5. vector-setall keep to 0???
            */
            
            // if(iou(boxes[order[i]], boxes[order[j]]) > threshold) {
            // 	keep[order[j]] = 0;
            // }
        }
    }

}

/* GPU implementation of NMS, for benchmarking */
void nms_gpu_src(float *boxes, int *order, int *keep, float threshold, int n) {

}

