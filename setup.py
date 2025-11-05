import re

from setuptools import find_packages, setup

with open("jungfrau_utils/__init__.py") as f:
    version = re.search(r'__version__ = "(.*?)"', f.read()).group(1)

setup(
    name="jungfrau_utils",
    version=version,
    description="",
    author="Paul Scherrer Institute",
    license="GNU GPLv3",
    packages=find_packages(exclude=["tests"]),
    zip_safe=False,
)
