from nms_serial import *
from nms_optimized import *
from utils import *
import time
import numpy as np
import random

def test_correctness(nmsfunc):
    testboxes, testprobs = read_binary_file("dataset/boxes.dat")
    testboxes = testboxes
    testprobs = np.asarray(testprobs)
    testthresholds = [0.1 * i for i in range(0, 10)]
    errors = 0.0
    for testthreshold in testthresholds:
        correctkeeps = nms_serial(testboxes, testprobs, testthreshold, "lowerleft")
        testkeeps = nmsfunc(testboxes, testprobs, testthreshold, "lowerleft")
        #print(errors)

        for i in range(len(correctkeeps)):
            if correctkeeps[i] != testkeeps[i]:
                errors += 1.0
    print("{}% error rate".format(str(100 * errors /(len(testboxes) * len(testthresholds)))))
    return True

def benchmark(nmsfunc):
    testboxes, testprobs = read_binary_file("dataset/boxes.dat")
    testprobs = np.asarray(testprobs)
    testthresholds = [random.random() for i in range(20)]

    starttime = time.time()
    for testthreshold in testthresholds:
        testkeeps = nmsfunc(testboxes, testprobs, testthreshold, "lowerleft")
    endtime = time.time() - starttime

    print(endtime)

if __name__ == "__main__":
    test_correctness(nms_c)
    test_correctness(nms_simd)
    test_correctness(nms_omp)

    benchmark(nms_serial)
    benchmark(nms_c)
    benchmark(nms_simd)
    benchmark(nms_omp)

