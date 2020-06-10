import setuptools

setuptools.setup(
    name="Vicon",
    version="1.0",
    install_requires=["GaitCore @ git+https://github.com/WPI-AIM/AIM_GaitCore.git"],
    py_modules=["Vicon", "ModelOutput", "Markers", "IMU", "ForcePlate", "EMG", "Devices", "Accel"]
)