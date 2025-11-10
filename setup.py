from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="crsm",
    version="0.1.0",
    author="CRSM Contributors",
    description="Continuous Reasoning State Model - A ~2B parameter autonomous language model with Mamba + MCTS architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pomilon/CRSM",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "einops>=0.6.0",
        "transformers>=4.30.0",
        "datasets>=2.10.0",
        "pytorch-lightning>=2.0.0",
        "tqdm>=4.60.0",
        "wandb>=0.13.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.39.0",
        "evaluate>=0.4.0",
        "pynvml>=11.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.20.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "flake8>=5.0.0",
        ],
        "colab": [
            "google-colab",
        ],
        "optional": [
            "mamba-ssm>=1.0.0",
            "state-spaces>=1.0.0",
        ],
    },
)