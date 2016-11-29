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
    # import ipdb; ipdb.set_trace()
    testboxes = testboxes
    testprobs = np.asarray(testprobs)
    testthresholds = [0.1 * i for i in range(0, 10)]
    errors = 0.0
    for testthreshold in testthresholds:
        correctkeeps = nms_serial(testboxes, testprobs, testthreshold, "lowerleft")
        testkeeps = nmsfunc(testboxes, testprobs, testthreshold, "lowerleft")

        for i in range(len(correctkeeps)):
            if correctkeeps[i] != testkeeps[i]:
                errors += 1.0
    print("{}% error rate, {} errors".format(str(100 * errors /(len(testboxes) * len(testthresholds))), errors))
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

def benchmark_full_dataset(nmsfunc, max_images=10000000, verbose=False):
    path = './data'
    n = 0
    total_time = 0
    running_avg = 0.0
    for filename in glob.glob(os.path.join(path, '*.txt')):
        testboxes, testprobs = read_text_file(filename)
        testboxes = map(bbox_center_to_diagonal, testboxes)

        testprobs = np.asarray(testprobs)

        testthreshold = 0.2
        # starttime = time.time()
        testkeeps, endtime = nmsfunc(testboxes, testprobs, testthreshold, "lowerleft", True)
        # endtime = time.time() - starttime
        n += 1
        running_avg = (running_avg*(n-1))/n + endtime/n
        if verbose:
            print(filename + ": " + str(endtime) + "  Running Average per file: " + str(running_avg))
        total_time += endtime
        if (n > max_images):
            break
    if verbose:
        print("Total time: " + str(total_time))
        print("Average Speed per file: " + str(running_avg))
    return (total_time, running_avg)

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

# def benchmark(nmsfunc):
#     testboxes, testprobs = read_binary_file("dataset/boxes.dat")
#     testprobs = np.asarray(testprobs)
#     testthresholds = [random.random() for i in range(20)]

#     starttime = time.time()
#     for testthreshold in testthresholds:
#         testkeeps = nmsfunc(testboxes, testprobs, testthreshold, "lowerleft")
#     endtime = time.time() - starttime

#     print(endtime)

def benchmark_multiple(functions, max_images=10000000, verbose=False):
    results = dict()
    fastest_function = None
    fastest_function_time = sys.maxsize
    for function_name in functions:
        if functions[function_name]:
            total_time, running_avg = benchmark_full_dataset(functions[function_name], max_images, verbose)
            results[function_name] = (total_time, running_avg)
            if running_avg < fastest_function_time:
                fastest_function_time = running_avg
                fastest_function = function_name
    print("")
    print("-----------------------------------------------------------------")
    for function_name in results:
        print(function_name + ": TOTAL TIME=" + str(results[function_name][0]) + " AVERAGE TIME PER IMAGE=" + str(results[function_name][1]))
    print("Fastest function is " + fastest_function + ".")
    for function_name in results:
        if function_name != fastest_function:
            multiple = "%.3f" % (results[function_name][1]/fastest_function_time)
            print(fastest_function + " is " + multiple + "x faster than " + function_name + " per image.")
    print("-----------------------------------------------------------------")


nms_functions = dict()

nms_functions["Serial_c"] = nms_c
nms_functions["Serial_py"] = nms_serial
nms_functions["Serial_Unordered"] = nms_c_unsorted_src
nms_functions["SIMD"] = nms_simd
nms_functions["OMP"] = nms_omp
nms_functions["OMP_alternate"] = nms_omp1
nms_functions["GPU"] = nms_gpu


if __name__ == "__main__":
    if len(sys.argv) > 1 and "-c" in sys.argv:
        print("c_unsorted")
        test_correctness(nms_c_unsorted_src)
        print("c_naive_serial")
        test_correctness(nms_c)
        print("c_simd")
        test_correctness(nms_simd)
        print("c_omp")
        test_correctness(nms_omp)
        print("c_omp1")
        test_correctness(nms_omp1)
        print("gpu")
        test_correctness(nms_gpu)

    # benchmark(nms_serial)
    # benchmark(nms_c)
    # benchmark(nms_simd)
    # benchmark(nms_omp)


    if len(sys.argv) > 1 and "-f" in sys.argv or"-fv" in sys.argv:
        if "-f" in sys.argv:
            nextElemIndex = sys.argv.index("-f") + 1
            verbose = False
        else:
            nextElemIndex = sys.argv.index("-fv") + 1
            verbose = True
        if len(sys.argv) > nextElemIndex:
            numImages = int(sys.argv[nextElemIndex])
            if numImages < 0:
                benchmark_multiple(nms_functions, verbose=verbose)
            else:
                benchmark_multiple(nms_functions, max_images=numImages, verbose=verbose)
        else:
            print("ERROR -f[v] [number of image]")
            print("full data benchmark expects an integer argument for the number of images to test, or -1 to test all images")
        # benchmark_full_dataset(nms_serial, 2)


    # if len(sys.argv) > 1 and "-c" in sys.argv:
    #     test_correctness_car_dataset(nms_c)

    # benchmark_and_check_accuracy_full_dataset(nms_c)
