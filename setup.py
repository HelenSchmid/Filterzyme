from setuptools import setup, find_packages
import os
import re


def read_version():
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'filtering_pipeline/__init__.py')
    with open(path, 'r') as fh:
        return re.search(r'__version__\s?=\s?[\'"](.+)[\'"]', fh.read()).group(1)


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='enzyme_filtering_pipeline',
      version=read_version(),
      description='',
      long_description=readme(),
      long_description_content_type='text/markdown',
      author='Helen Schmid',
      author_email='schmid.helen2@gmail.com',
      url='https://github.com/HelenSchmid/EnzymeStructuralFiltering',
      license='GPL3',
      project_urls={
          "Bug Tracker": "https://github.com/HelenSchmid/EnzymeStructuralFiltering/issues",
          "Documentation": "https://github.com/HelenSchmid/EnzymeStructuralFiltering",
          "Source Code": "https://github.com/HelenSchmid/EnzymeStructuralFiltering",
      },
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
      ],
      keywords='util',
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'filtering_pipeline = filtering_pipeline.__main__:main'
          ]
      },
      install_requires=['pandas', 'numpy', 'pybiomart'],
      python_requires='>=3.6',
      data_files=[("", ["LICENSE"])]
      )