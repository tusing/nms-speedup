#include <stdlib.h>
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

/* Source: http://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection */
float lowerleft_iou(float *box1, float *box2) {
    // determine the (x, y)-coordinates of the intersection rectangle
    float xA = max(box1[0], box2[0]);
    float yA = max(box1[1], box2[1]);
    float xB = min(box1[2], box2[2]);
    float yB = min(box1[3], box2[3]);
    
    // compute area of intersection, prediction, and ground-truth rectangles
    float interArea = (xB - xA + 1) * (yB - yA + 1);
    float box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1);
    float box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1);
    
    // compute and return iou
    return interArea / (box1Area + box2Area - interArea);
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
void* nms_c_src(float *boxes, int *order, int *keep, float threshold, int n) {
    for(int i=0; i<n; i++) {
        if(!keep[order[i]]) {
            continue;
        }
        for(int j=i+1; j<n; j++) {
            if(iou(boxes + 4*order[i], boxes + 4*order[j]) > threshold) {
                keep[order[j]] = 0;
            }
        }
    }
    return keep;  
    
}

/* Vectorized implementation of NMS, for benchmarking */
void* nms_simd_src(float *boxes, int *order, int *keep, float threshold, int n) {
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
    return keep;
}

/* GPU implementation of NMS, for benchmarking */
void* nms_gpu_src(float *boxes, int *order, int *keep, float threshold, int n) {
    return 0;
}

