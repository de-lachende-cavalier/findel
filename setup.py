from setuptools import setup, find_packages

setup(
    name="findel",
    version="0.1.0",
    packages=find_packages(exclude=["tests", "data", "notebooks"]),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.2.0",
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "statsmodels>=0.14.0",
        "yfinance>=0.2.0",
        "pyfolio>=0.9.2",
        "empyrical>=0.5.5",
        "ta>=0.10.0",
        "tqdm>=4.65.0",
    ],
    description="Finance-specific deep learning approaches inspired by physics-based methodologies",
    keywords="finance, deep learning, machine learning, neural networks",
    python_requires=">=3.11",
)
