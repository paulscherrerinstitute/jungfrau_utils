from setuptools import setup, Extension

setup(
    name='jungfrau_utils',
    version='0.2.2',
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
            ],
        ),
    ],
)
