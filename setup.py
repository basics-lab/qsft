from distutils.core import setup

setup(
    packages=['awesomeRNA', 'awesomeRNA.params'],

    package_dir={
        'awesomeRNA': './awesomeRNA'},

    package_data={
        'awesomeRNA': ['params/*.par']},

    install_requires=[
        'numpy',
        'ViennaRNA']
)
