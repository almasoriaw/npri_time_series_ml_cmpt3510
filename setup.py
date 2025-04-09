from setuptools import find_packages, setup

setup(
    name="npri_time_series",
    version="1.0.0",
    description="Time series analysis and forecasting of pollutant release trends from the NPRI dataset",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/NPRI_time_series_ML_CMPT3510",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "plotly",
        "joblib"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
