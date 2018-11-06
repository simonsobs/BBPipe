#!/usr/bin/env python
"""
Lightweight B-modes pipeline constructor
Based on DESC's ceci by F. Lanusse, J. Zuntz and others
"""
from setuptools import setup

setup(
    name='bb_pipe',
    version='0.0.5',
    description='Lightweight pipeline constructor for B-modes',
#    url='https://github.com/LSSTDESC/bb_pipe',
#    maintainer='Joe Zuntz',
#    license='MIT',
#    classifiers=[
#        'Intended Audience :: Developers',
#        'Intended Audience :: Science/Research',
#        'License :: OSI Approved :: MIT License',
#        'Programming Language :: Python :: 3.6',
#    ],
    packages=['bb_pipe', 'bb_pipe.sites', 'bb_pipe_example'],
    entry_points={
        'console_scripts':['bb_pipe=bb_pipe.main:main']
    },
    install_requires=['pyyaml','parsl<0.6.0']
)
