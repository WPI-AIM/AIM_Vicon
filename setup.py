import setuptools
from setuptools import  find_packages

setuptools.setup(
    name="Vicon",
    version="1.4",
    install_requires=[
        "GaitCore @ git+https://github.com/WPI-AIM/AIM_GaitCore.git",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib"
    ],
    packages=find_packages()
)
