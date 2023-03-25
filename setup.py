from setuptools import setup, find_packages

setup(
    name = 'policygradient',
    version = 0.1,
    packages = find_packages(),
    install_requires = ['pip >= 19.3.1', 'numpy', 'tensorflow==2.11.1', 'gym', 'falcon'],
    include_package_data = True
)
