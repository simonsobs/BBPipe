#!/usr/bin/env python
"""
B-modes pipeline constructor
Based on DESC's ceci by F. Lanusse, J. Zuntz and others
https://github.com/LSSTDESC/ceci
"""
from setuptools import setup

setup(
    name='BBPipe',
    version='0.0.5',
    description='Lightweight pipeline constructor for B-modes',
    url='https://github.com/simonsobs/BBPipe',
    maintainer='David Alonso',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    packages=['bbpipe', 'bbpipe.sites'],
    entry_points={
        'console_scripts':['bbpipe=bbpipe.main:main']
    },
    install_requires=['pyyaml','parsl<0.6.0','cwlgen']
)
