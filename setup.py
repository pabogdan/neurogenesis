from setuptools import setup, find_packages

setup(
    name='spinnaker_neurogenesis',
    version='0.1.0',
    packages=find_packages(),
    url='pabogdan.com',
    license="GNU GPLv3.0",
    author='Petrut Antoniu Bogdan',
    author_email='petrut.bogdan@manchester.ac.uk',
    description='Simulating neurogenesis on SpiNNaker',
    # Requirements
    dependency_links=[],

    install_requires=["numpy", "scipy", "brian2", "pynn==0.7.5", "spynnaker>= 3.0.0, < 4.0.0"],
    classifiers=[
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",

        "Programming Language :: Python :: 2.7"
    ]
)
