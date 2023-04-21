import os

import toml
from hyfi import HyFI, about, global_config

from ._version import __version__

# Read and parse pyproject.toml
current_dir = os.path.dirname(os.path.abspath(__file__))
pyproject_path = os.path.join(current_dir, "project.toml")

with open(pyproject_path) as f:
    pyproject_data = toml.load(f)

# Extract package information from pyproject.toml
project_data = pyproject_data.get("metadata", {})
about.name = project_data.get("name", "package name")
about.author = project_data.get("authors", ["Author name"])[0]
about.description = project_data.get("description", "package description")
about.homepage = project_data.get("homepage", "https://package.homepage")
about.version = __version__
global_config.hyfi_package_config_path = "pkg://thematos.conf"


def get_version() -> str:
    """This is the cli function of the package"""
    return __version__
