""" Setup script
"""

from setuptools import setup

setup(name='div_detection',
      version='0.2',
      description='Cell-division detection',
      author='Andrew Berger',
      author_email='andbberger@gmail.com',
      url='https://github.com/rueberger/division_detection',
      py_modules=['division_detection'],
      install_requires=[
          'numpy',
          'scipy',
          'keras',
          'h5py',
          'pathos',
          'matplotlib',
          'scikit-image',
          'scikit-learn'
      ])
