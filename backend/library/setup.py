from setuptools import setup, find_packages

setup(
    name="alivee",
    version="0.3.0",
    description="Lifelong Learning Library for Digital Entities",
    author="Antigravity",
    packages=find_packages(include=['alivee', 'alivee.*']),
    install_requires=[
        "torch",
        "numpy",
        "psutil"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
