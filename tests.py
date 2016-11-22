from nms_serial import *
from nms_optimized import *
from utils import *
import time
import numpy as np
import random
import os
import glob
import sys


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

def test_correctness_car_dataset(nmsfunc):
    '''
    Test a single data file in the folder data, even this can take >5 minutes
    because the serial implementation is so slow
    '''
    path = './data'
    for filename in glob.glob(os.path.join(path, '*.txt')):
        testboxes, testprobs = read_text_file(filename)
        testboxes = map(bbox_center_to_diagonal, testboxes)

        testprobs = np.asarray(testprobs)
        testthresholds = [0.0, 0.2, 0.451325, 0.6, 0.8]
        print("Testing " + filename)
        for testthreshold in testthresholds:
            print("...")
            correctkeeps = nms_serial(testboxes, testprobs, testthreshold, "lowerleft")
            print("...")
            testkeeps = nmsfunc(testboxes, testprobs, testthreshold, "lowerleft")
            print("...")
            for i in range(len(correctkeeps)):
                if correctkeeps[i] != testkeeps[i]:
                    print("Failure error")
                    return False
        print("Success!")
        return True
    print("No file found in data")
    return False

def benchmark_full_dataset(nmsfunc):
    path = './data'
    n = 0
    total_time = 0
    running_avg = 0.0
    for filename in glob.glob(os.path.join(path, '*.txt')):
        testboxes, testprobs = read_text_file(filename)
        testboxes = map(bbox_center_to_diagonal, testboxes)

        testprobs = np.asarray(testprobs)
        testthresholds = [0.0, 0.2, 0.451325, 0.6, 0.8]

        starttime = time.time()
        for testthreshold in testthresholds:
            testkeeps = nmsfunc(testboxes, testprobs, testthreshold, "lowerleft")
        endtime = time.time() - starttime
        n += 1
        running_avg = (running_avg*(n-1))/n + endtime/n
        print(filename + ": " + str(endtime) + "  Running Average per file: " + str(running_avg))
        total_time += endtime
    print("Total time: " + str(total_time))
    print("Average Speed per file: " + str(running_avg))

def benchmark_and_check_accuracy_full_dataset(nmsfunc):
    path = './data'
    res_sha = time.strftime("D%d-M%m_h%I-m%M-s%S")
    res_path = './results/' + res_sha
    os.mkdir(res_path)
    res_path = res_path + "/data"
    os.mkdir(res_path)
    n = 0
    total_time = 0
    running_avg = 0.0
    testthreshold = 0.2
    for filename in glob.glob(os.path.join(path, '*.txt')):
        classes = read_text_file_by_object(filename)
        endtime = 0
        f = open(os.path.join(res_path, os.path.basename(filename)), "w")
        for object_class in classes:
            testboxes = map(bbox_center_to_diagonal, classes[object_class][0])
            testprobs = np.asarray(classes[object_class][1])
            starttime = time.time()
            keeps = nmsfunc(testboxes, testprobs, testthreshold, "lowerleft")
            endtime += time.time() - starttime
            for i in range(len(keeps)):
                if keeps[i]:
                    f.write(classes[object_class][2][i])
        n += 1
        running_avg = (running_avg*(n-1))/n + endtime/n
        print(filename + ": " + str(endtime) + "  Running Average per file: " + str(running_avg))
        total_time += endtime
        f.close() 
    print("Total time: " + str(total_time))
    print("Average Speed per file: " + str(running_avg))

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


    if len(sys.argv) > 1 and "-f" in sys.argv:
        benchmark_full_dataset(nms_c)

    if len(sys.argv) > 1 and "-c" in sys.argv:
        test_correctness_car_dataset(nms_c)

    # benchmark_and_check_accuracy_full_dataset(nms_c)
