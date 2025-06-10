"""Setup script for Smart Image Cropper."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip()
                    and not line.startswith("#")]

setup(
    name="smart-image-cropper",
    version="1.0.0",
    author="Giulio Manuzzi",
    author_email="giuliomanuzzi@gmail.com",
    description="An intelligent image cropping library that creates smart collages",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/giumanuz/image_cropper",
    project_urls={
        "Bug Tracker": "https://github.com/giumanuz/image_cropper/issues",
        "Documentation": "https://github.com/giumanuz/image_cropper#readme",
        "Source Code": "https://github.com/giumanuz/image_cropper",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="image processing, cropping, collage, computer vision, opencv",
)
