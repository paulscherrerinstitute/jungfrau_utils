#!/usr/bin/env python3

import argparse
import os
import re
import subprocess


def main():
    default_branch = "main"
    branch = subprocess.check_output("git rev-parse --abbrev-ref HEAD", shell=True).decode().strip()
    if branch != default_branch:
        print(f"Aborting, not on '{default_branch}' branch.")
        return

    version_filepath = os.path.join(os.path.basename(os.path.dirname(__file__)), "__init__.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("level", type=str, choices=["patch", "minor", "major"])
    args = parser.parse_args()

    with open(version_filepath) as f:
        file_content = f.read()

    version = re.search(r'__version__ = "(.*?)"', file_content).group(1)
    major, minor, patch = map(int, version.split(sep="."))

    if args.level == "patch":
        patch += 1
    elif args.level == "minor":
        minor += 1
        patch = 0
    elif args.level == "major":
        major += 1
        minor = 0
        patch = 0

    new_version = f"{major}.{minor}.{patch}"

    with open(version_filepath, "w") as f:
        f.write(re.sub(r'__version__ = "(.*?)"', f'__version__ = "{new_version}"', file_content))

    os.system(f"git commit {version_filepath} -m 'Updating for version {new_version}'")
    os.system(f"git tag -a {new_version} -m 'Release {new_version}'")
    os.system("git push --follow-tags")


if __name__ == "__main__":
    main()
