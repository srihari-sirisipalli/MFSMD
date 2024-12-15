from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="MFSMD",  # The name of your package/project
    version="0.1.0",  # Initial version
    author="Sri Hari Sirisipalli",
    author_email="sriharisirisipalli0@gmail.com",
    description="Machine Fault Detection and Monitoring System (MFSMD)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/srihari-sirisipalli/MFSMD",  # Your GitHub repository URL
    packages=find_packages(where="src"),  # Automatically discover all packages in src/
    package_dir={"": "src"},  # Root folder is src
    include_package_data=True,  # Include files specified in MANIFEST.in
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Adjust the license if different
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Minimum supported Python version
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "statsmodels>=0.12.0",
        "scikit-image>=0.18.0",
        "pyqt5>=5.15.0",
        "streamlit>=1.0.0",
        "loguru>=0.5.3",
    ],  # Project dependencies
    extras_require={
        "dev": ["pytest", "flake8", "black"],  # Development dependencies
    },
    entry_points={
        "console_scripts": [
            "mfsmd-monitor=src.frontend.app:main",  # Example CLI command
        ],
    },
)
