from distutils.core import setup, Extension
from distutils.command.install_data import install_data
from compiler.pycodegen import compileFile
import glob
import distutils
import distutils.sysconfig
import distutils.core
import os
import py2exe

VERSION='1.2.0'

LONG_DESCRIPTION="""\
Fast Artificial Neural Network Library implements multilayer
artificial neural networks with support for both fully connected
and sparsely connected networks. It includes a framework for easy 
handling of training data sets. It is easy to use, versatile, well 
documented, and fast. 
"""

class smart_install_data(install_data):
    """
    override default distutils install_data, so we can copy
    files directly, without splitting into modules, scripts,
    packages, and extensions.
    """
    def run(self):
        # need to change self.install_dir to the actual library dir

        install_cmd = self.get_finalized_command('install')
        self.install_dir = getattr(install_cmd, 'install_lib')
        return install_data.run(self)

def hunt_files(root, which):
    return glob.glob(os.path.join(root, which))

data_files = []

# add sources
data_files = data_files + [['', ['fann.py', '__init__.py']]]

# add dll and swig output
compileFile('libfann.py')
data_files = data_files + [['', ['libfann.pyc', '_libfann.pyd']]]

# add examples
data_files = data_files + [['examples', hunt_files('examples', '*.py')]]

# add examples datasets
data_files = data_files + [['examples/datasets', hunt_files('../benchmarks/datasets', 'mushroom*')]]
data_files = data_files + [['examples/datasets', hunt_files('../examples', 'xor.data')]]

setup(
    name='pyfann',
    description='Fast Artificial Neural Network Library (fann)',
    long_description=LONG_DESCRIPTION,
    version=VERSION,
    author='Steffen Nissen',
    author_email='lukesky@diku.dk',
    maintainer='Gil Megidish',
    maintainer_email='gil@megidish.net',
    url='http://sourceforge.net/projects/fann/',
    platforms='WIN32',
    license='GNU LESSER GENERAL PUBLIC LICENSE (LGPL)',
    data_files=data_files,
    cmdclass={'install_data': smart_install_data},
    extra_path='pyfann'
)

