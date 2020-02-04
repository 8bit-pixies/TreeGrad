from setuptools import setup, find_packages
import os

readme_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md')
try:
    from m2r import parse_from_file
    readme = parse_from_file(readme_file)
except ImportError:
    # m2r may not be installed
    with open(readme_file, 'r') as f:
        readme = f.read()

ISRELEASED = True
MAJOR = 1
MINOR = 1
MICRO = 0
VERSION = f"{MAJOR}.{MINOR}.{MICRO}"

setup_dir = os.path.abspath(os.path.dirname(__file__))
VFNAME = 'treegrad/version.py'



def write_version_py(filename=os.path.join(setup_dir, VFNAME)):
    """
    Generate the version.py file automatically upon install only
    """
    version = VERSION
    if not ISRELEASED:
        version += '.dev'

    a = open(filename, 'w')
    file_content = "\n".join(["",
                              "# THIS FILE IS GENERATED FROM SETUP.PY",
                              "version = '%(version)s'",
                              "isrelease = '%(isrelease)s'"])

    a.write(file_content % {'version': VERSION,
                            'isrelease': str(ISRELEASED)})
    a.close()


write_version_py()

NAME = "treegrad"
DESCRIPTION = ("transfer parameters from lightgbm to differentiable decision trees!")
AUTHOR = 'Chapman Siu'

install_requires = ['autograd', 'sklearn', 'lightgbm']
                    
extras_require = {'development': ['sphinx>=1.6.6',
                                  'sphinxcontrib-napoleon>=0.6.1',
                                  'pandoc>=1.0.2',
                                  'nbsphinx>=0.3.3',
                                  'nose2>=0.7.4',
                                  'nose2_html_report>=0.6.0',
                                  'coverage>=4.5.1',
                                  'awscli>=1.15.26',
                                  'flake8>=3.5.0',
                                  'm2r']
                 }
extras_require['complete'] = sorted(set(sum(extras_require.values(),[])))

setup(
    url="http://github.com/chappers/TreeGrad",
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=readme,
    author=AUTHOR,
    author_email="chpmn.siu@gmail.com",
    include_package_data=True,
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.5", # supporting type hints
    zip_safe=False  # force install as source
)
