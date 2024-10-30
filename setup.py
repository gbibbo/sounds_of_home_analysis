# setup.py

from setuptools import setup, find_packages

setup(
    name='sounds_of_home_analysis',
    version='0.1.0',
    author='Gabriel Bibbó',
    author_email='g.bibbo@surrey.ac.uk',
    description='Tools for visualizing sound events detected in the "Sounds of Home" experiment.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/gbibbo/sounds_of_home_analysis',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'matplotlib>=3.0',
        'numpy>=1.17',
        'tqdm>=4.0',
        'pandas>=1.0',
        'requests>=2.0',
        'beautifulsoup4>=4.0',
        'lxml>=4.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
