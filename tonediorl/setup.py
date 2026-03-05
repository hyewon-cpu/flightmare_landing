import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

setup(
    name='tonedio_baselines',
    version='0.0.1',
    author='Heejun Shin',
    author_email='heejun@tonedio.io',
    description='Flightmare: A Quadrotor Simulator.',
    long_description='',
    install_requires=['gymnasium', 'ruamel.yaml',
                      'numpy', 'stable-baselines3', 'opencv-python'],
    packages=['tonedio_baselines'],
)

