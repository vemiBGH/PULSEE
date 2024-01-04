from setuptools import setup, find_packages

setup(name="pulsee",
      packages=find_packages(where='src'),
      package_dir={"": "src"}
      )
