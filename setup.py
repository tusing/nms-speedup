from distutils.core import setup, Extension

DESC = ("Optimized NMS functions")


nms_module = Extension('nms',
                    sources = ['nmsModule.c'])

setup (name = 'Optimized NMS',
       version = '0.1',
       description = DESC,
       author="CS 194 Students",
       url="https://github.com/tusing/nms-speedup",
       license="MIT",
       
       ext_modules = [nms_module])
