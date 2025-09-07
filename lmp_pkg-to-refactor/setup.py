from setuptools import setup, find_packages

setup(
    name="lmp_apps",
    version="1.0.0",
    description="A Virtual Bioequivalence Simulation Package",
    author="Duy Nguyen",
    
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "numba",
        "pandas",
        "scikit-learn",
        "joblib",
        "matplotlib",
        "seaborn",
        "statsmodels",
        "ipython"
    ],
    package_data={
        'lmp_apps': ['data/cfd/*', 'data/lung/*'],
    },
    include_package_data=True,
)
