"""Python setup.py for diploma_thesis package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("diploma_thesis", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="diploma_thesis",
    version=read("diploma_thesis", "VERSION"),
    description="Awesome diploma_thesis created by yura-hb",
    url="https://github.com/yura-hb/diploma-thesis/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="yura-hb",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["diploma_thesis = diploma_thesis.__main__:main"]
    },
    extras_require={"test": read_requirements("requirements-test.txt")},
)
