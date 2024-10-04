from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()

setup(
    name='ronja',
    version='0.1',
    packages=find_packages() + ['prisoner_pettingzoo_env'],
    package_data={'prisoner_pettingzoo_env': ['*']},
    install_requires=parse_requirements('requirements.txt'),
    entry_points={
        'console_scripts': [
            'ronja=ronja.scripts.train:main',
        ],
    },
)