from setuptools import find_packages, setup

setup(name='torch_control',
    author="Feng Gao",
    author_email="feng.gao220@gmail.com",
    packages=find_packages(include="torch_control"),
    version='0.0.0',
    install_requires=[
        'hydra-core', #==1.3.2',
        'gym', #==0.26.2',
        'meshcat', #==0.3.2',
        'numpy', #==1.26.1',
        'omegaconf', #==2.3.0',
        'pygame', #==2.4.0',
        'torch', #==2.1.0',
        'wandb', #==0.16.0'
        'matplotlib',
        'path',
        'pandas',
        'scipy',
        'transformers'
        ]
)
