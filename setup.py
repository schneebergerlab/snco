from setuptools import setup

if __name__ == '__main__':
    setup(
        name='snco',
        version='0.3.1',
        description=(
            'crossover mapping for single cell/nucleus sequencing data'
        ),
        author='Matthew Parker',
        entry_points={
            'console_scripts': [
                'snco = snco.main:main',
                'sneqtl = snco.main:sneqtl'
            ]
        },
        packages=[
            'snco',
        ],
        scripts=['scripts/syri_vcf_to_stardiploid.py',
                 'scripts/collapse_ha_specific_alns.py',
                 'scripts/single_cell_gene_counts.py'],
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
            'scikit-learn',
            'seaborn', # only for sneqtl plotting, may remove in future
        ],
    )
