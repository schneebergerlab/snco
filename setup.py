from setuptools import setup, find_packages

setup(
    name='snco',
    version='0.6.1',
    description=(
        'crossover mapping for single cell/nucleus sequencing data'
    ),
    author='Matthew Parker',
    entry_points={
        'console_scripts': [
            'snco = snco.main:main',
        ]
    },
    packages=find_packages(),
    scripts=['scripts/syri_vcf_to_stardiploid.py',
             'scripts/collapse_ha_specific_alns.py'],
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'torch',
        'pomegranate>=1.0',
        'click',
        'joblib',
        'pysam',
        'matplotlib',
    ],
    extras_require={
        'test': ['pytest']
    },
    tests_require=['pytest'],
)
