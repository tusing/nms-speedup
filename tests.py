from nms_serial import *
from nms_optimized import *
from utils import *
import time
import numpy as np

def test_correctness(nmsfunc):
    testboxes, testprobs = read_binary_file("dataset/boxes.dat")

    testprobs = np.asarray(testprobs)
    testthresholds = [0.0, 0.2, 0.451325, 0.6, 0.8, 1.0]

    for testthreshold in testthresholds:
        correctkeeps = nms_serial(testboxes, testprobs, testthreshold, "lowerleft")
        testkeeps = nmsfunc(testboxes, testprobs, testthreshold, "lowerleft")
        for i in range(len(correctkeeps)):
            if correctkeeps[i] != testkeeps[i]:
                print("Failure error")
                return False
    print("Success")
    return True

def benchmark(nmsfunc):
    testboxes, testprobs = read_binary_file("dataset/boxes.dat")
    testprobs = np.asarray(testprobs)
    testthresholds = [0.0, 0.2, 0.451325, 0.6, 0.8, 1.0]

    starttime = time.time()
    for testthreshold in testthresholds:
        testkeeps = nmsfunc(testboxes, testprobs, testthreshold, "lowerleft")
    endtime = time.time() - starttime

    print(endtime)
    
if __name__ == "__main__":
    test_correctness(nms_c)
    test_correctness(nms_omp)
    
    benchmark(nms_serial)
    benchmark(nms_c)
    benchmark(nms_omp)
