#!/usr/bin/env python
from distutils.core import setup, Extension
#from glob import glob

VERSION='1.2.0'

LONG_DESCRIPTION="""\
Fast Artificial Neural Network Library implements multilayer
artificial neural networks with support for both fully connected
and sparsely connected networks. It includes a framework for easy 
handling of training data sets. It is easy to use, versatile, well 
documented, and fast. 
"""

module1 = Extension(
    '_libfann', 
    sources = ['libfann.i', 'fann_helper.c'], 
    libraries = ['fann'],
    #extra_objects = glob('../src/fann*.o'),
    )

setup(
    name='pyfann',
    version=VERSION,
    description='Fast Artificial Neural Network Library (fann)',
    long_description=LONG_DESCRIPTION,
    author='Steffen Nissen',
    author_email='lukesky@diku.dk',
    maintainer='Gil Megidish',
    maintainer_email='gil@megidish.net',
    url='http://sourceforge.net/projects/fann/',
    license='GNU LESSER GENERAL PUBLIC LICENSE (LGPL)',
    platforms='UNIX',

    ext_modules = [module1],
    py_modules = ['libfann', 'fann']
    )

