from setuptools import setup, find_packages

setup(
    name='spinnaker_structural_plasticity',
    version='1.0.0',
    packages=find_packages(),
    url='https://github.com/pabogdan/neurogenesis',
    license="GNU GPLv3.0",
    author='Petrut Antoniu Bogdan',
    author_email='petrut.bogdan@manchester.ac.uk',
    description='Simulating Structural plasticity on SpiNNaker',
    # Requirements
    dependency_links=[],

    install_requires=["numpy",
                      "scipy",
                      "brian2",
                      "spynnaker",
                      "matplotlib"],
    classifiers=[
        "Development Status :: 3 - Alpha",

        "Intended Audience :: Science/Research",

        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",

        "Programming Language :: Python :: 3"
        "Programming Language :: Python :: 3.7"
        
        "Topic :: Scientific/Engineering",
    ]
)
