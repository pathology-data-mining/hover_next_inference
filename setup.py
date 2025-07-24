from setuptools import setup, find_packages

# Read the content of README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="hover_next_inference",
    version="0.1",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    long_description=long_description,
    long_description_content_type='text/markdown',
    entry_points={'console_scripts': ['hover-next-inference=inference.__main__:main']},
)
