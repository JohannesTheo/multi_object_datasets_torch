"""Installation script for setuptools."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup

EXTRA_PACKAGES = {

}

setup(
    name='multi_object_datasets_torch',
    version='1.0.0',
    author='Johannes Theodoridis',
    description=('Multi-object image datasets with'
                 'ground-truth segmentation masks and generative factors.'),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    keywords=['datasets', 'machine learning', 'representation learning'],
    url='https://github.com/JohannesTheo/multi_object_datasets_torch',
    download_url='https://github.com/JohannesTheo/multi_object_datasets_torch.git',
    packages=['multi_object_datasets_torch'],
    package_dir={'multi_object_datasets_torch': '.'},
    install_requires=[
        'h5py',
        'tqdm',
        'numpy',
        'gsutil',
        'multi-object-datasets @ git+https://github.com/deepmind/multi_object_datasets.git@master'
    ],
    extras_require={
        'pytorch': ['torch', 'torchvision'],
    },
    python_requires=">=3.8",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
