# HoVer-NeXt Inference Documentation

Welcome to the HoVer-NeXt inference documentation!

## Documentation Files

- **[API.md](API.md)** - Comprehensive API documentation for developers
  - Installation and setup
  - Quick start examples
  - Module reference
  - Advanced usage patterns
  - Custom extensions guide

## Main Repository Documentation

- **[README.md](../README.md)** - User guide with installation and usage instructions
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** - Guidelines for contributors
- **[REFACTORING_SUMMARY.md](../REFACTORING_SUMMARY.md)** - Summary of recent improvements

## Quick Links

### For Users
- [Quick Start Guide](../README.md#quick-start)
- [Installation Options](../README.md#setup)
- [Command-Line Arguments](../README.md#command-line-arguments)
- [Example Configurations](../example_config.sh)

### For Developers
- [API Reference](API.md#api-reference)
- [Core Modules](API.md#core-modules)
- [Custom Extensions](API.md#custom-extensions)
- [Source Code Documentation](../src/inference/)

### For Contributors
- [Development Setup](../CONTRIBUTING.md#development-environment-setup)
- [Code Style](../CONTRIBUTING.md#code-style)
- [Testing](../CONTRIBUTING.md#testing-your-changes)
- [Submitting Changes](../CONTRIBUTING.md#submitting-changes)

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/pathology-data-mining/hover_next_inference/issues)
- **Discussions**: Create an issue for questions
- **Paper**: [OpenReview Publication](https://openreview.net/pdf?id=3vmB43oqIO)

## Code Structure

```
hover_next_inference/
├── src/inference/          # Main package source code
│   ├── __main__.py        # CLI entry point
│   ├── inference.py       # Core inference pipeline
│   ├── post_process.py    # Post-processing pipeline
│   ├── data_utils.py      # Dataset classes
│   ├── augmentations.py   # Color augmentation
│   ├── spatial_augmenter.py # Geometric augmentation
│   ├── multi_head_unet.py # Model architecture
│   ├── viz_utils.py       # Visualization utilities
│   └── constants.py       # Configuration constants
├── docs/                   # Documentation
│   ├── README.md          # This file
│   └── API.md             # API documentation
├── main.py                # Convenience script
├── setup.py               # Package configuration
└── README.md              # User guide
```

## Documentation Status

All major modules now include:
- ✅ Module-level docstrings
- ✅ Function and class docstrings
- ✅ Parameter and return value documentation
- ✅ Usage examples
- ✅ Inline comments for complex algorithms
- ✅ Comprehensive API reference

## Contributing to Documentation

We welcome improvements to documentation! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

When adding new features, please:
1. Add docstrings to all new functions and classes
2. Update the API.md if adding public APIs
3. Add examples for complex usage patterns
4. Update README.md if user-facing features change
