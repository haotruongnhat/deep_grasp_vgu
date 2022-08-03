# -*- coding: utf-8 -*-

"""
Setup for deep_grasp_vgu
"""
import os
from glob import glob

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['dougsm_helpers','ggcnn', 'models', 'utils'],
    package_dir={'': 'src'},
)

setup(**d)
