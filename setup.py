from setuptools import setup, find_packages

setup(
    name="thermography_toolbox",
    version="0.2.0", 
    packages=find_packages(where="."),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "torch>=1.10.0",
        "PyWavelets>=1.3.0",  
        "matplotlib>=3.5.0"   
    ],
    python_requires=">=3.8",
    
    # Metadata
    author="Konrad Jaworski",
    description="Thermography analysis toolbox",
    license="MIT",
    keywords="thermography ndt wavelet pct ppt dmd",
    
    # Important for correct package discovery
    package_dir={"": "."},
    include_package_data=True,
)