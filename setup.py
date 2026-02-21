from setuptools import setup, find_packages

setup(
    name="stc-capsnet",
    version="1.0.0",
    author="Taiba Majid Wani, Syed Asif Ahmad Qadri, Irene Amerini",
    author_email="majid@diag.uniroma1.it",
    description="STC-CapsNet: Detecting Audio Deepfakes with Spatio-Temporal Convolutions and Capsule Networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/CapsuleNetworks/STC-CapsNet",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchaudio>=0.12.0",
        "numpy>=1.21.0",
        "librosa>=0.9.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "noisereduce>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
