from setuptools import setup, find_packages

setup(
    name="synapnet-edge",
    version="0.1.0",
    description="Efficient long-context AI inference on consumer hardware — quantized hybrid SSM+attention+memory architecture reproducible in 25 minutes on a MacBook",
    author="Vineetha Vallish Kumar",
    license="MIT",
    url="https://github.com/vineetha00/SynapNet-Edge",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": ["psutil", "pyyaml", "tqdm", "scipy"],
        "mlx": ["mlx>=0.12.0", "mlx-lm>=0.0.1"],
        "full": ["datasets>=2.18.0", "transformers>=4.40.0", "psutil", "pyyaml", "tqdm"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
