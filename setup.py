from setuptools import setup


setup(
    name='snco',
    version='0.0pre',
    description=(
        'crossover mapping for single cell/nucleus sequencing data'
    ),
    author='Matthew Parker',
    entry_points={
        'console_scripts': [
            'snco = snco.main:main',
        ]
    },
    packages=[
        'snco',
    ],
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'pomegranate>=1.0',
        'click',
        'click-log',
        'joblib',
        'pysam',
    ],
)