"""Tests import errors."""

import importlib
import sys
from unittest.mock import patch

import pyimcom.imdestripe
import pytest


def test_missing_furry_parakeet():
    """What happens if furry_parakeet is missing."""

    with patch.dict(sys.modules, {"furry_parakeet": None, "furry_parakeet.pyimcom_interface": None}):
        with pytest.warns(UserWarning):
            # we have to try importing, so telling ruff not to remove this.
            importlib.reload(pyimcom.imdestripe)  # noqa: F401
    importlib.reload(pyimcom.imdestripe)  # noqa: F401

    # load pyimcom.lakernel with routines, but then put it back.
    with patch.dict(sys.modules, {"furry_parakeet": None, "furry_parakeet.pyimcom_croutines": None}):
        importlib.reload(pyimcom.lakernel)  # noqa: F401
    importlib.reload(pyimcom.lakernel)  # noqa: F401
