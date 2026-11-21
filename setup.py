from setuptools import Extension, setup

setup(ext_modules=[Extension("alexpy._native", sources=["alexpy/_native.c"])])
