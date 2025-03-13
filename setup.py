from setuptools import setup, find_packages

with open("llada/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llada",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Large Language Diffusion with mAsking (LLaDA) implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llada",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "tqdm>=4.62.0",
    ],
) 