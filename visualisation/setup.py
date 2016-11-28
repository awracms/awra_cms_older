from setuptools import setup,find_packages

setup(name='awrams.visualisation',
      namespace_packages=['awrams'],
      packages=['awrams.visualisation'],
      version='0.1',
      description='Spatial and Temporal Data Visualisation for AWRA Modelling System',
      url='https://gitlab.bom.gov.au/awra/awrams_cm.git',
      author='awrams team',
      author_email='awrams@bom.gov.au',
      license='MIT',
      zip_safe=False,
      include_package_data=False,
      setup_requires=['nose>=1.3.3'],
      # install_requires=[
      #     'numpy==1.9.3', # this fails
      #     'gdal==1.10.0'  # requires .so already built
      # ],
      test_suite='nose.collector',
      tests_require=['nose']
      )
