from setuptools import setup,find_packages

setup(name='awrams.calibration',
      namespace_packages=['awrams'],
      packages=['awrams.calibration','awrams.calibration.objectives'],
      version='0.1',
      description='Calibration components for AWRA Modelling System',
      url='https://gitlab.bom.gov.au/awra/awrams_cm.git',
      author='awrams team',
      author_email='awrams@bom.gov.au',
      license='MIT',
      zip_safe=False,
      include_package_data=False,
      setup_requires=['nose>=1.3.3'],
      test_suite='nose.collector',
      tests_require=['nose']
      )
