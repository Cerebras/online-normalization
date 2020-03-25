"""
Released under BSD 3-Clause License,
Copyright (c) 2019 Cerebras Systems Inc.
All rights reserved.
"""
from setuptools import setup, find_packages


def readme():
    with open('README.md', 'r') as f:
        return f.read()


setup(name='online_norm_pytorch',
      version='0.1',
      description='PyTorch Implementation of Online Normalization',
      long_description=readme(),
      long_description_content_type="text/markdown",
      keywords='online normalization pytorch',
      url='https://github.com/Cerebras/online-normalization',
      author='Cerebras Systems',
      author_email='info@cerebras.net',
      packages=find_packages(),
      install_requires=[
          'torch',
      ],
      classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
