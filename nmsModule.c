#include <stdlib.h>
#include <math.h>
#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

/* BICHEN'S ORIGINAL IOU CODE (wrong? negative values when both boxes same)
*/
float iou(float *box1, float *box2) {
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

