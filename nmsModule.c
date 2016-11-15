#include <Python.h>


/* Scalar naive implementation of NMS, for benchmarking */
static PyObject* nms_c_src(PyObject* self, PyObject* args) {
    /* Convert python args to C equivalents, execute NMS */
    return Py_BuildValue("");
}

/* Scalar naive implementation of NMS, for benchmarking */
static PyObject* nms_simd_src(PyObject* self, PyObject* args) {
    /* Convert python args to C equivalents, execute NMS */
    return Py_BuildValue("");
}

/* Scalar naive implementation of NMS, for benchmarking */
static PyObject* nms_gpu_src(PyObject* self, PyObject* args) {
    /* Convert python args to C equivalents, execute NMS */
    return Py_BuildValue("");
}


/* Bind Python names to our c functions */
static PyMethodDef nmsModule_methods[] = {
    {"nms_c_src", nms_c_src, METH_VARARGS},
    {"nms_simd_src", nms_simd_src, METH_VARARGS},
    {"nms_gpu_src", nms_gpu_src, METH_VARARGS},
    {NULL, NULL}};

static struct PyModuleDef nms = {
    PyModuleDef_HEAD_INIT,
    "nms",
    "",
    -1,
    nmsModule_methods
};

PyMODINIT_FUNC PyInit_nms(void) {
    return PyModule_Create(&nms);
}

