"""File name test functions. Right now tests error handling."""

import pytest
from pyimcom.coadd import InImage
from pyimcom.compress.compressutils import CompressedOutput, ReadFile


def test_fname_errs():
    """Test exceptions: file name errors."""

    with pytest.raises(ValueError, match=r"unrecognized file type"):
        CompressedOutput("unknown_file")

    with pytest.raises(ValueError, match=r"Scheme notascheme not supported"):
        ReadFile("notascheme://testonly/test.test")

    with pytest.raises(ValueError, match=r"psf_filename: unknown format"):
        InImage.psf_filename("we_dont_have_a_format_called_this", 1989)
