import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "robot_utils",
    version = "0.0.1",
    author = "Christopher 'ckt' Tomaszewski",
    author_email = "christomaszewski@gmail.com",
    description = ("A library of common utilities used in robotics"),
    license = "BSD",
    keywords = "robotics utilities",
    url = "https://github.com/christomaszewski/robot_utils.git",
    packages=['robot_utils', 'tests', 'examples'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)