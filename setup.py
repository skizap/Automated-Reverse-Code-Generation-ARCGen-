#!/usr/bin/env python3
"""
Setup script for ARCGen V2
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README_V2.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')

setup(
    name="arcgen-v2",
    version="2.0.0",
    author="Enhanced by AI Assistant (Original by Tammy)",
    author_email="",
    description="Enhanced Automated Reverse Code Generation for code optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/arcgen-v2",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Information Technology",
        "Topic :: Security",
        "Topic :: Software Development :: Code Generators",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "arcgen=arcgen_v2:main",
            "arcgen-v2=arcgen_v2:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md", "*.txt"],
    },
    keywords=[
        "code-generation", "ai", "optimization", "code-analysis", "automation"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/arcgen-v2/issues",
        "Source": "https://github.com/your-username/arcgen-v2",
        "Documentation": "https://github.com/your-username/arcgen-v2/blob/main/README_V2.md",
    },
) 