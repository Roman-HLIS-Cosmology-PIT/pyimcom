"""File name test functions. Right now tests error handling."""

import numpy as np
from pyimcom.compress.compressutils import CompressedOutput, ReadFile

def test_fname_errs():
    """Test exceptions: file name errors."""

    try:
        c = CompressedOutput("unknown_file")
    except Exception as e:
        assert str(e) == "unrecognized file type"

    try:
        c = ReadFile("notascheme://testonly/test.test")
    except ValueError as e:
        assert str(e) == "Scheme notascheme not supported"
