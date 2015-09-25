import sys
try:
    from setuptools import setup
except ImportError:
    print("Error, please install setuptools by running: \n{0}".format("wget https://bootstrap.pypa.io/ez_setup.py -O - |sudo python"))
    sys.exit(1)


classifiers = """\
Environment :: Console
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Healthcare Industry
License :: Public Domain
Programming Language :: Python
Topic :: Scientific/Engineering :: Medical Science Apps.
Topic :: Scientific/Engineering :: Mathematics
"""

scripts = [
    'analyze = imagequality.analyze:main',
    'analyze%d = imagequality.analyze:main' % sys.version_info[:1],
    'analyze-%d.%d = imagequality.analyze:main' % sys.version_info[:2],
]

__version__ = "0.5.1"
__url__ = "http://www.fda.gov/AboutFDA/CentersOffices/OfficeofMedicalProductsandTobacco/CDRH/CDRHOffices/ucm299950.htm"

setup_args = dict(name = 'imagequality'
                  ,version = __version__,
                  packages = ['imagequality'],
                  entry_points = {'console_scripts':scripts},
                  author = 'Colin Versteeg and Lucretiu Popescu',
                  author_email = 'cverstee@gmail.com',
                  classifiers = classifiers.splitlines(),
                  url = __url__,
                  install_requires=['numpy','matplotlib'],
                  use_2to3 = True
)

def main():
    setup(**setup_args)

if __name__ == '__main__':
    main()