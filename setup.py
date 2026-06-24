"""Package setup for EchoGuard — real-time deepfake and voice clone detector."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="echoguard",
    version="0.1.0",
    author="vrutulpatel",
    author_email="vrutulpatel25@gmail.com",
    description="Real-time, privacy-first AI deepfake and voice clone detector",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vrutulpatel/echoguard",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video :: Analysis",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "echoguard=src.main:main",
        ],
    },
)
