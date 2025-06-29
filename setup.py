from setuptools import setup, find_packages

setup(
    name="thermography_toolbox",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "torch"
    ],
    python_requires=">=3.7",
)