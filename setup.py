from setuptools import setup, find_packages

VERSION = "0.0.5"
DESCRIPTION = "deeplut, a python lib to train look up tables (LUTs) natively"
LONG_DESCRIPTION = "A Python library that aims to provide a flexible, extendible, lightening fast, and easy-to-use framework to train look-up tables (LUT) deep neural networks from scratch."

# Setting up
setup(
    name="deeplut",
    version=VERSION,
    author="Mohamed Adel",
    author_email="a-moadel@outlook.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],
    keywords=["python", "first package"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
