from setuptools import setup, find_packages

setup(
    name="thermography_toolbox",
    version="0.3.0",  # Bumped version for new PyTorch features
    packages=find_packages(where="."),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "torch>=1.10.0",
        "PyWavelets>=1.3.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.62.0",          
        "opencv-python>=4.5.0",   
        "h5py>=3.6.0",           
        "numba>=0.55.0"           
    ],
    extras_require={
        'gpu': ["cudatoolkit>=11.3"],  
        'full': [
            "pycuda>=2021.1",
            "cupy-cuda11x>=10.0.0",  
            "memory_profiler>=0.60.0"
        ],
        'docs': [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0"
        ]
    },
    python_requires=">=3.8",
    
    # Metadata
    author="Konrad Jaworski",
    description="Advanced Thermography Analysis Toolbox with PyTorch acceleration",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    keywords="thermography ndt pytorch wavelet pct ppt dmd",
    url="https://github.com/konrad-jaworski/Thermography_toolbox/tree/main",
    
    # Important for correct package discovery
    package_dir={"": "."},
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)