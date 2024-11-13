import setuptools
import sys


__name__ = "pyhrv"
__version__ = "0.4.2"
__author__ = "Pedro Gomes (Original Author), Constantino Álvarez Casado, Contributors"
__original_author__ = "Pedro Gomes (Plux Biosignals - Portugal)"
__original_url__ = "https://github.com/PGomes92/pyhrv"
__fork_author__ = "Constantino Álvarez Casado (University of Oulu)"
__fork_url__ = "https://github.com/Arritmic/pyhrv2"
__fork_email__ = "constantino.alvarezcasado@oulu.fi"
__description__ = "An updated and optimized fork of pyhrv with new features."
# __long_description__ = ""


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Check Python version compatibility
if sys.version_info < (3, 11):
    sys.exit("This package requires Python 3.11 or higher. Please upgrade your Python version.")

# Create setup
setuptools.setup(
    name=__name__,
    version=__version__,
    author=__author__,
    author_email=__fork_email__,
    description=__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.11',
    url=__fork_url__,
    keywords=['Heart Rate Variability', 'HRV', 'Healthcare', 'Signal Processing', 'AI'],
    # The package dependencies
    install_requires=[
        'biosppy>=2.0.0',
        'matplotlib>=3.5.0',
        'numpy>=1.21.0',
        'scipy>=1.14.0',
        'nolds>=0.6.1',
        'spectrum>=0.8.1',
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,  # Include additional files specified in MANIFEST.in
    package_data={
        "pyhrv": ["files/*", "README.md", "references.txt", "files/quickstart/*"]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    project_urls={
        "Original Repository": "https://github.com/PGomes92/pyhrv",
        "Original Author": f"Author: {__original_author__}",
        "Fork Repository": __fork_url__,
        "Bug Tracker": "https://github.com/Arritmic/pyhrv2/issues",
        "Documentation": "https://pyhrv2.readthedocs.io/en/latest/",
        "Source Code": __fork_url__,
    },
)