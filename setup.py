from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="MFSMD",  # Your module name
    version="0.1.0",  # Initial version
    author="Your Name",
    author_email="your.email@example.com",
    description="Machine Fault Detection and Monitoring System",
    long_description=long_description,
    long_description_content_type="text/markdown",  # Optional (use README.md format)
    url="https://github.com/yourusername/MFSMD",  # Replace with your project URL
    packages=find_packages(),  # Automatically find all packages in the project
    include_package_data=True,  # Include files from MANIFEST.in
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Adjust license if needed
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Minimum Python version required
    install_requires=[
        "numpy",
        "scipy",
        "scikit-image",
        "statsmodels",
        "matplotlib",
        "pyqt5",
        "pandas",
        "streamlit"
    ],  # List all required packages here
    extras_require={  # Optional extras
        "dev": ["pytest", "flake8"],
    },
    entry_points={  # Optional: for command-line tools
        "console_scripts": [
            "mfsmd-app = MFSMD.frontend.app:main",  # Example: run the frontend app
        ],
    },
)
