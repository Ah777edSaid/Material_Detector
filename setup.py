from setuptools import setup, find_packages

setup(
    name="MobileNet",
    version="0.1",
    packages=find_packages(),  # Automatically finds all packages in the project
    install_requires=[          # List any dependencies here
        "tensorflow",
        "opencv-python",
        "numpy",
        "requests",
    ],
)
