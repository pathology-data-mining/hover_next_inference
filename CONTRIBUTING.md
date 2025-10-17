# Contributing to HoVer-NeXt Inference

Thank you for your interest in contributing to HoVer-NeXt! This document provides guidelines and information for contributors.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/hover_next_inference.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Set up the development environment (see below)

## Development Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate hovernext

# Install PyTorch with CUDA support
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118

# Install the package in editable mode
pip install -e .
```

## Code Style

- Follow PEP 8 guidelines for Python code
- Use descriptive variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular

## Testing Your Changes

Before submitting a pull request:

1. Test your changes with the examples in `example_config.sh`
2. Ensure the package builds successfully: `python -m build`
3. Verify the help message works: `python3 main.py --help`
4. Test with different input types (WSI, NPY, images)

## Submitting Changes

1. Commit your changes with clear, descriptive commit messages
2. Push to your fork: `git push origin feature/your-feature-name`
3. Open a pull request against the main repository
4. Describe your changes and their purpose in the PR description

## Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- CUDA version (if applicable)
- Full error message and traceback
- Steps to reproduce the issue
- Example input files (if possible)

## Feature Requests

We welcome feature requests! Please open an issue and describe:

- The use case for the feature
- How it would benefit users
- Any implementation ideas you have

## Code of Conduct

- Be respectful and constructive in all interactions
- Welcome newcomers and help them get started
- Focus on what is best for the community

## Questions?

If you have questions about contributing, feel free to open an issue or contact the maintainers.

Thank you for contributing to HoVer-NeXt!
