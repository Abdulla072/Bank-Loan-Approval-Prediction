from setuptools import setup, find_packages

setup(
    name="loan_approval",
    version="0.0.1",
    author="Abdulla",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)