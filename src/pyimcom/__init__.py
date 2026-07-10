import warnings

from asdf.exceptions import AsdfConversionWarning, AsdfPackageVersionWarning

from ._version import __version__  # noqa: F401

# disable some asdf warnings
warnings.filterwarnings("ignore", category=AsdfConversionWarning)
warnings.filterwarnings("ignore", category=AsdfPackageVersionWarning)
