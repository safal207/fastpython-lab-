from setuptools import Extension, setup

setup(ext_modules=[
    Extension("alexpy._native", sources=["alexpy/_native.c"]),
    Extension("alexpy._memprobe", sources=["alexpy/_memprobe.c"]),
    Extension("alexpy._buffer_native", sources=["alexpy/_buffer_native.c"]),
    Extension("alexpy._qa_native", sources=["alexpy/_qa_native.c"]),
])
