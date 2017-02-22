from setuptools import setup

setup(name="mandate_calculator",
      version='1.0',
      description="Daniel Rona's mandate calculator implemented in Python.",
      url='https://gitlab.com/k.balint/mandate_calculator',
      author='Balint Kubik',
      author_email='kubikbalint@gmail.com',
      license='MIT',
      packages=['mandate_calculator'],
      install_requires=['numpy'],
      test_suite='nose.collector',
      tests_require=['nose',
                     'pandas'],
      zip_safe=False)