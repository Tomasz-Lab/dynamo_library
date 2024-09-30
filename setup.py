from setuptools import setup, find_packages

setup(
    name='dynamo',
    version='0.1',
    description='A Python library designed to analyze the temporal variability of the gut microbiome.',
    author='Zuzanna Karwowska, Paulina Dziwak',
    packages=find_packages(include=['dynamo']),
    python_requires='>=3.10',
    install_requires=[
        'numpy~=1.26.4',
        'pandas~=2.2.2',
        'scipy~=1.14.0',
        'matplotlib~=3.9.1',
        'scikit-learn~=1.5.1',
        'scikit-bio~=0.6.2',
        'seaborn~=0.13.2',
        'statsmodels~=0.14.2',
        'arch~=7.0.0',
        'librosa~=0.10.2.post1',
        'wheel~=0.43.0'
    ],
    test_suite='tests',
)
