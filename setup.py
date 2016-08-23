from setuptools import setup, find_packages

setup(
    name='spinnaker_neurogenesis',
    version='2016.0.1',
    packages=find_packages(),
    url='pabogdan.com',
    license='',
    author='Petrut Antoniu Bogdan',
    author_email='pab@cs.man.ac.uk',
    description='Simulating neurogenesis on SpiNNaker',
    # Requirements
    dependency_links=[],

    install_requires=["numpy", "scipy", "brian2", "pynn==0.7.5", "spynnaker>=2016.1.1"],
    classifiers=[
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",

        "Programming Language :: Python :: 2.7"
    ]
)
