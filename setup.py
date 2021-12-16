#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#############################################
# File Name: setup.py
# Author: qhduan
# Mail: mail@qhduan.com
# Created Time: 2020-12-16 09:00:00
#############################################

import os
from setuptools import setup, find_packages

current_dir = os.path.realpath(os.path.dirname(__file__))

ON_RTD = os.environ.get('READTHEDOCS') == 'True'
if not ON_RTD:
    INSTALL_REQUIRES = open(os.path.join(
        current_dir,
        'requirements.txt'
    )).read().split('\n')
else:
    INSTALL_REQUIRES = []

VERSION = '0.0.2'

setup(
    name='video-matting',
    version=VERSION,
    keywords=('pip', 'video'),
    description='video tool',
    long_description='video tool',
    license='GPLv3',
    url='https://github.com/deepdialog/rvm',
    author='qhduan',
    author_email='mail@qhduan.com',
    packages=find_packages(),
    include_package_data=True,
    platforms='any',
    install_requires=INSTALL_REQUIRES,
    entry_points={
        'console_scripts': [
            'video-matting = rvm.__main__:cmd',
        ],
    },
)
