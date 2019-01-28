from setuptools import setup, Extension

setup(
    name='jungfrau_utils',
    version='0.2.1',
    description='',
    author='Paul Scherrer Institute',
    license='GNU GPLv3',
    packages=['jungfrau_utils', 'jungfrau_utils.scripts', 'jungfrau_utils.plot'],
    ext_modules=[
        Extension('libcorrections', sources=['jungfrau_utils/src/corrections.c']),
    ],
    # managed by conda
    #install_requires=[
    #    'pyzmq',
    #],
)
