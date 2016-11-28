from setuptools import setup,find_packages

setup(name='awrams.utils',
      namespace_packages=['awrams'],
      packages=['awrams.utils','awrams.utils.messaging','awrams.utils.io','awrams.utils.ts',
                'awrams.utils.nodegraph'],
      version='0.1',
      description='utilities for AWRAL',
      url='https://gitlab.bom.gov.au/awra/awrams_cm.git',
      author='awrams team',
      author_email='awrams@bom.gov.au',
      license='MIT',
      zip_safe=False,
      include_package_data=True,
      setup_requires=['nose>=1.3.3'],
      # install_requires=[
      #     'numpy==1.9.3', # this fails
      #     'gdal==1.10.0'  # requires .so already built
      # ],
      # test_suite='tests',
      test_suite='nose.collector',
      tests_require=['nose'],
      )
