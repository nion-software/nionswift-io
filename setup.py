# -*- coding: utf-8 -*-

"""
To upload to PyPI, PyPI test, or a local server:
python setup.py bdist_wheel upload -r <server_identifier>
"""

import setuptools
import os

setuptools.setup(
    name="nionswift-io",
    version="0.0.1",
    author="Nion Software",
    author_email="swift@nion.com",
    description="IO handlers for NionSwift.",
    url="https://github.com/nion-software/nionswift-io",
    packages=["nionswift_plugin.DM_IO", "nionswift_plugin.TIFF_IO"],
    install_requires=['nionswift'],
    license='GPLv3',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.5",
    ],
    include_package_data=True,
    python_requires='~=3.5',
)