#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="multilevel",
    use_scm_version=True,
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'torch',
        'jupyter',
        'ipython',
        'scipy',
        'autograd',
        'numdifftools',
        'pandas',
        'unzip'
    ],
    setup_requires=[
        'setuptools_scm',
    ],
)
