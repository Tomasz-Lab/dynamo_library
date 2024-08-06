from setuptools import setup, find_packages

setup(
    name='dynamo',
    version='0.1',
    description='A Python library designed to analyze the temporal variability of the gut microbiome.',
    author='Zuzanna Karwowska',
    packages=find_packages(include=['dynamo']),
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
