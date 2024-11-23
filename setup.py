from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='predictive model to analyze youtube comments',
    author='ronilpatil',
    license='MIT',
)
# pip install -e . --use-pep517 (create pyproject.toml and hit this command to install local package in editable mode)
