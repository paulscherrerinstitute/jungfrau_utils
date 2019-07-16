import re

from setuptools import setup, Extension

with open("jungfrau_utils/__init__.py") as f:
    version = re.search(r'__version__ = "(.*?)"', f.read()).group(1)

setup(
    name='jungfrau_utils',
    version=version,
    description='',
    author='Paul Scherrer Institute',
    license='GNU GPLv3',
    packages=['jungfrau_utils', 'jungfrau_utils.scripts', 'jungfrau_utils.plot'],
    ext_modules=[
        Extension(
            'libcorrections',
            sources=['jungfrau_utils/src/corrections.c'],
            extra_compile_args=[
                '--std=c99',
                '-mtune=core-avx2',
                '-shared',
                '-O3',
            ],
        ),
    ],
)
