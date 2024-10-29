from setuptools import setup

if __name__ == '__main__':
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
        scripts=['scripts/syri_vcf_to_stardiploid.py'],
        install_requires=[
            'numpy',
            'scipy',
            'pandas',
            'torch',
            'pomegranate>=1.0',
            'click',
            'joblib',
            'pysam',
            'matplotlib' # todo: make this dependency optional
        ],
    )