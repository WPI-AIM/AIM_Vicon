import setuptools
from setuptools import  find_packages

setuptools.setup(
    name="Vicon",
    version="1.0",
    install_requires=["GaitCore @ git+https://github.com/nag92/AIM_GaitCore.git",
        "numpy",
        "scipy",
        "pandas"
    ],
    packages=find_packages()
)
