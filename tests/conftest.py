import os

import pytest


@pytest.fixture(scope="session")
def is_ci():
    return os.environ.get("CI", "false").lower() == "true"
