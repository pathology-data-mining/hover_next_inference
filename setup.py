from setuptools import setup, find_packages

# Read the content of README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open('requirements.txt', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="hover_next_inference",
    version="0.1.0",
    author="Elias Baumann, Josef Lorenz Rumberger",
    author_email="",
    description="Fast and efficient nuclei segmentation and classification pipeline",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/pathology-data-mining/hover_next_inference",
    project_urls={
        "Bug Tracker": "https://github.com/pathology-data-mining/hover_next_inference/issues",
        "Documentation": "https://github.com/pathology-data-mining/hover_next_inference#readme",
        "Source Code": "https://github.com/pathology-data-mining/hover_next_inference",
        "Publication": "https://openreview.net/pdf?id=3vmB43oqIO",
    },
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'hover-next-inference=inference.__main__:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="nuclei segmentation classification pathology histology deep-learning",
)
