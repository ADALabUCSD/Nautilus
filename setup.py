 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from distutils.core import setup
from setuptools import find_packages
import os
import sys

this = os.path.dirname(__file__)

packages = find_packages()
assert packages

# read version from the package file.
with (open(os.path.join(this, "nautilus/__init__.py"), "r")) as f:
    line = [_ for _ in [_.strip("\r\n ") for _ in f.readlines()] if _.startswith("__version__")]
    if len(line) > 0:
        version_str = line[0].split("=")[1].strip('" ')

README = os.path.join(os.getcwd(), "README.md")
with open(README) as f:
    long_description = f.read()
    start_pos = long_description.find("## Introduction")
    if start_pos >= 0:
        long_description = long_description[start_pos:]

install_requires = [
    "tf2onnx",
    "gurobipy>=9.1.2",
    "networkx",
    "numpy"
]
setup(
    name="nautilus",
    version=version_str,
    description="Optimized System for Deep Transfer Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache Software License",
    author="",
    author_email="",
    url="",
    packages=packages,
    include_package_data=True,
    install_requires=install_requires,
    extras_require={
        "tests": ["flake8", "pytest", "coverage", "pre-commit"],
        "benchmark": ["memory-profiler", "psutil"],
    },
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires=">=3.5",
)
