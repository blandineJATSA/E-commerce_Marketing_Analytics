# setup.py
from setuptools import setup, find_packages

setup(
    name="e-commerce-mlops",
    version="1.0.0",
    description="MLOps Pipeline for E-commerce Marketing Analytics",
    author="Votre Nom",
    author_email="votre.email@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.3",
        "numpy>=1.24.3",
        "scikit-learn>=1.3.0",
        "mlflow>=2.5.0",
        "airflow>=2.6.3",
        "fastapi>=0.100.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
)
