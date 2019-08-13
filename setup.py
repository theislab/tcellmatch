from setuptools import setup, find_packages

author = 'David S. Fischer, Yihan Wu'
author_email='david.fischer@helmholtz-muenchen.de'
description="Predicting T-cell to epitope specificity."

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='tcellmatch',
    author=author,
    author_email=author_email,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'tensorflow>=1.13',
        'tensorflow-probability>=0.5',
        'numpy>=1.14.0',
        'scipy',
        'pandas',
        'patsy',
    ],
    extras_require={
        'tensorflow_gpu': [
            "tensorflow-gpu",
            "tensorflow-probability-gpu",
        ],
        'docs': [
            'sphinx',
            'sphinx-autodoc-typehints',
            'sphinx_rtd_theme',
            'jinja2',
            'docutils',
        ],
    },
    version="0.1.0",
)
