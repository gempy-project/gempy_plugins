# setup.py for gempy_viewer. Requierements are numpy and matplotlib
from os import path

from setuptools import setup, find_packages

setup(
    name='gempy_plugins',
    packages=find_packages(),
    url='',
    license='EUPL-1.2',
    author='Miguel de la Varga', 
    author_email="miguel@terranigma-solutions.com",
    description='Extra plugins for the geological modeling package GemPy',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: GIS',
        'Programming Language :: Python :: 3.10'
    ],
    python_requires='>=3.10',
    extras_require={
        'geological-simulation': ['gempy', 'numpy', 'scipy', 'torch'],
        'geological-simulation-viz': ['matplotlib', 'pyvista'],
    },
    setup_requires=['setuptools_scm'],
    use_scm_version={
            "root"            : ".",
            "relative_to"     : __file__,
            "write_to"        : path.join("gempy_plugins", "_version.py"),
            "fallback_version": "3.0.0"
    },
)
