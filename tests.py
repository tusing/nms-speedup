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
    ''' Test a single data file in the folder data, even this can take >5 minutes because the serial implementation is so slow. '''
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
        testkeeps, endtime = nmsfunc(testboxes, testprobs, testthreshold, "lowerleft", True)

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
    print("Function\tTotal Time\tAvg. Time Per Image")
    print("{:<16}{:<16}{:<16}".format("Function Name", "Total Time", "Avg. Time/Image"))
    print("-----------------------------------------------------------------")
    for function_name in results:
        print("{:<16}{:<16}{:<16}".format(function_name, str(results[function_name][0]), str(results[function_name][1])))
    print("Fastest function is " + fastest_function + ".")
    print("")
    print("")
    print("{:<16}{:<16}{:<16}".format("Function Name", "C Speedup", "Python Speedup"))
    print("-----------------------------------------------------------------")
    for function_name in results:
        multipleC = "%.3f" % (results["Serial_c"][1]/results[function_name][1])
        multiplePy = "%.3f" % (results["Serial_py"][1]/results[function_name][1])
        print("{:<16}{:<16}{:<16}".format(function_name, multipleC, multiplePy))
    print("-----------------------------------------------------------------")

nms_functions = dict()
nms_functions["Accurate serial python"] = nms_serial
nms_functions["Accurate serial c"] = nms_c
nms_functions["Inaccurate unordered"] = nms_c_unsorted_src
nms_functions["Accurate SIMD OMP"] = nms_simd
nms_functions["Accurate OMP"] = nms_omp
nms_functions["Inaccurate OMP"] = nms_omp1
nms_functions["Inaccurate GPU"] = nms_gpu

if __name__ == "__main__":
    if len(sys.argv) > 1 and "-c" in sys.argv:
        for n, f in nms_functions.iteritems():
            print(n)
            test_correctness(f)

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
