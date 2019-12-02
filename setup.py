from distutils.core import setup


setup(
    name='pdpiper',
    version='0.1.0',
    description='Pipelines for pandas dataframes.',
    author='Jonas Busk',
    author_email='jonasbusk@gmail.com',
    url='https://github.com/jonasbusk/pdpiper',
    packages=['pdpiper'],
    license='MIT',
    install_requires=[
        'numpy >= 1.16.4',
        'pandas >= 0.24.2',
    ],
)
