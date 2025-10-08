"""
Compression tools for PyIMCOM outputs.

Classes
-------
CompressedOutput
    Main class for compresseing a FITS file.

Functions
---------
_parser
    File name parser; only needed for file names with regular expressions.
ReadFile
    Stand-alone function to read a compressed FITS file.
test
    Long version of the test.
test1
    Short version of the test.

"""

import re
import sys
import time
from urllib.parse import urlparse

import numpy as np
from astropy.io import fits

from ..config import Config

# specific compression tools we need
from .i24 import i24compress, i24decompress


class CompressedOutput:
    """Class for compressing pyimcom output files.

    Parameters
    ----------
    fname : str
        File name for uncompressed file.
    format : str or None, optional
        Compression format.
    extraargs : dict, optional
        Extra arguments for astropy.io.fits.

    Attributes
    ----------
    gzip : bool
        Is gzipped?
    ftype : str
        File type. Currently the only option is 'fits'.
    cprstype : str
        Compression type (for a full file, not currently used).
    origfile : str
        Original file name (when opened)
    hdul : astropy.io.fits.HDUList
        HDU List of information (initially a deep copy of the input file).
    cfg : pyimcom.config.Config
        Configuration file used to generate this image.

    Methods
    -------
    __init__
        Constructor.
    compress_2d_image
        Wrapper for 2d image compression (staticmethod).
    decompress_2d_image
        Wrapper for 2d image decompression (staticmethod).
    compress_layer
        Compresses a layer.
    decompress
        Decompresses the whole file.
    recompress
        Recompress previously compressed layers.
    to_file
        Saves to a file.
    close
        Close associated file.

    Notes
    -----
    At some point, we may add file formats other than FITS, but not yet.

    This object is big since it stores a deep copy of the whole file.
    Therefore, it is not a good idea to open lots of compressed files at once.

    """

    def __init__(self, fname, format=None, extraargs={}):
        self.origfile = fname

        # figure out what type of file it is, and if it is gzipped
        self.gzip = False
        if fname[-3:] == ".gz":
            self.gzip = True
        if format is None:
            pref = fname[:-3] if self.gzip else fname

            # right now supports fits files
            if pref[-5:] == ".fits":
                self.ftype = "fits"
                self.hdul = fits.open(fname, mode="readonly", decompress_in_memory=True, **extraargs)
                if "CPRSTYPE" in self.hdul[0].header:
                    self.cprstype = self.hdul[0].header["CPRSTYPE"]
                else:
                    self.cprstype = ""
                    self.hdul[0].header["CPRSTYPE"] = ""
                self.cfg = Config(fname, inmode="block")

            else:
                raise Exception("unrecognized file type")

    @staticmethod
    def compress_2d_image(im, scheme, pars):
        """Wrapper to compress a 2D image.

        Parameters
        ----------
        im : np.array
            2D image to be compressed.
        scheme : str
            Name of the compression scheme.
        pars : dict
            Parameters to be passed to the compression algorithm.

        Returns
        -------
        imout : np.array
            The compressed 2D image, same shape as `im`.
        ovflow : astropy.io.fits.BinTableHDU
            A table of values that overflowed the quantization range for the
            compression scheme. Returns None if not used.

        """

        if scheme[:3] == "I24":
            imout, ovflow = i24compress(im, scheme, pars)
            return imout, ovflow

        # unrecognized scheme or NULL
        return np.copy(im), None

    @staticmethod
    def decompress_2d_image(im, scheme, pars, overflow=None):
        """Wrapper to decompress a 2D image; overflow table used for some formats.

        Parameters
        ----------
        im : np.array
            The compressed image.
        scheme : str
            Name of the compression scheme.
        pars : dict
            Parameters to be passed to the compression algorithm.
        overflow : astropy.io.fits.BinTableHDU or None, optional
            (If used.) A table of values that overflowed the quantization range for the
            compression scheme. Returns None if not used.

        Returns
        -------
        imout : np.array
            The de-compressed 2D image, same shape as `im`.

        """

        if scheme[:3] == "I24":
            return i24decompress(im, scheme, pars, overflow=overflow)

        # unrecognized scheme or NULL
        return np.copy(im)

    def compress_layer(self, layerid, scheme=None, pars={}):
        """Compresses the given layerid with the indicated scheme.

        The pars is a dictionary of parameters that go with that scheme.
        It must be in a format that supports a FITS header.
        If scheme is None, then the algorithm will re-compress the layer in the same way was
        done previously (if it was compressed before), otherwise it will do the NULL compression.

        Parameters
        ----------
        layerid : int
            Number of the layer to be compressed.
        scheme : str or None, optional
            Name of the compression scheme. Defaults to None (no compression).
        pars : dict, optional
            Parameters to be passed to the compression algorithm.

        Returns
        -------
        None

        """

        # this failure to write the EXTNAME shouldn't happen, but just in case
        if layerid >= 16**4:
            return

        # make a blank table if there isn't compression information yet
        if "CPRESS" not in [hdu.name for hdu in self.hdul]:
            hdu = fits.TableHDU.from_columns([fits.Column(name="text", format="A512", array=[], ascii=True)])
            hdu.name = "CPRESS"
            self.hdul.append(hdu)
        # now get compression information
        rows = self.hdul["CPRESS"].data["text"].tolist()

        # None means we should check if this data has been compressed
        # before and use that method.
        if scheme is None:
            compressiondict = {}
            for kv in rows:
                layer_, key_, val_ = kv.strip().split(":")
                key_ = key_.strip()
                val_ = val_.strip()
                if int(layer_, 0x10) == layerid:
                    compressiondict[key_] = val_
            if "SCHEME" in compressiondict:
                # this was done before
                # re-compress without adding new keywords
                cd_data, cd_overflow = CompressedOutput.compress_2d_image(
                    self.hdul[0].data[0, layerid, :, :], compressiondict["SCHEME"], compressiondict
                )
                self.hdul[0].data[0, layerid, :, :] = 0
                newhdu = fits.ImageHDU(cd_data)
                for p in compressiondict:
                    newhdu.header[p] = compressiondict[p]
                newhdu.name = f"HSHX{layerid:04X}"
                self.hdul.append(newhdu)
                cd_overflow.name = f"HSHV{layerid:04X}"
                self.hdul.append(cd_overflow)

                # print('re-compressed', compressiondict)
                return

            # we will do the null compression in this case
            scheme = "NULL"

        cd_data, cd_overflow = CompressedOutput.compress_2d_image(
            self.hdul[0].data[0, layerid, :, :], scheme, pars
        )
        self.hdul[0].data[0, layerid, :, :] = 0
        newhdu = fits.ImageHDU(cd_data)
        for p in pars:
            newhdu.header[p] = pars[p]
            rows += [f"{layerid:04X}:{p:8s}:{str(pars[p]):s}"]
        newhdu.header["SCHEME"] = scheme
        rows += ["{:04X}:{:8s}:{:s}".format(layerid, "SCHEME", scheme)]
        newhdu.name = f"HSHX{layerid:04X}"
        self.hdul.append(newhdu)
        cd_overflow.name = f"HSHV{layerid:04X}"
        self.hdul.append(cd_overflow)

        # which HDU has the compression data?
        j_ = -1
        for j in range(len(self.hdul)):
            if self.hdul[j].name == "CPRESS":
                j_ = j
        if j_ == -1:
            raise Exception("Can't find CPRESS: this shouldn't happen.")
        # now overwrite that one
        hdu = fits.TableHDU.from_columns([fits.Column(name="text", format="A512", array=rows, ascii=True)])
        hdu.name = "CPRESS"
        self.hdul[j_] = hdu

    def decompress(self):
        """Decompresses all the layers that were compressed by compress_layer."""

        j = 0
        while j < len(self.hdul):
            if self.hdul[j].name[:4] == "HSHX":
                layer_target = int(self.hdul[j].name[-4:], 0x10)
                self.hdul[0].data[0, layer_target, :, :] = CompressedOutput.decompress_2d_image(
                    self.hdul[j].data,
                    self.hdul[j].header["SCHEME"],
                    self.hdul[j].header,
                    overflow=self.hdul["HSHV" + self.hdul[j].name[-4:]],
                )
                del self.hdul[j]
            else:
                # we only increment j if the we didn't remove anything, since if we removed an HDU then
                # what was hdul[j+1] is now hdul[j]
                j = j + 1

        # remove the overflow tables
        j = 0
        while j < len(self.hdul):
            if self.hdul[j].name[:4] == "HSHV":
                del self.hdul[j]
            else:
                j = j + 1

    def recompress(self):
        """Recompresses all the layers that were previously compressed by compress_layer."""

        # if this wasn't compressed before, nothing to do
        if "CPRESS" not in [hdu.name for hdu in self.hdul]:
            return

        # figure out which layers were previously compressed
        nlayer = np.shape(self.hdul[0].data)[-3]
        wascompressed = np.zeros(nlayer, dtype=bool)
        for compressnote in self.hdul["CPRESS"].data["text"].tolist():
            ilayer = int(compressnote.split(":")[0], 16)  # which layer this referred to
            wascompressed[ilayer] = True
        # print(wascompressed)
        for ilayer in range(nlayer):
            if wascompressed[ilayer]:
                self.compress_layer(ilayer)

    def to_file(self, fname, overwrite=False):
        """
        Saves to a file.

        Parameters
        ----------
        fname : str
            File name.
        overwrite : bool, optional
            Whether to overwrite the file.

        Returns
        -------
        None

        """

        self.hdul.writeto(fname, overwrite=overwrite)

    def close(self):
        """Closes the associated file."""
        self.hdul.close()

    def __enter__(self):
        """Entry method."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Has context manager automatically close."""
        self.close()
        return False  # do not suppress exception


def _parser(fname):
    """
    Re-formats a file name containing a regular expression.

    Regular expressions are separated with the ^ character and contain a row and column index,
    followed by the suffix. For example::

        >>> _parser('hello_world/Q_02_31.fits') # no ^, regular file name
        'hello_world/Q_02_31.fits'
        >>> _parser('hello_world/Row{1:2d}/Q_{0:02d}_{1:02d}^_02_31.fits') # FITS file
        'hello_world/Row31/Q_02_31.fits'
        >>> _parser('hello_world/Row{1:2d}/Q_{0:02d}_{1:02d}^_02_12.fits.gz') # gzipped; suffix is copied over
        'hello_world/Row12/Q_02_12.fits.gz'

    This is useful if the files are not all in the same directory.

    Parameters
    ----------
    fname : str or str-like
        Regular file name (not including ^) or regular expression.

    Returns
    -------
    str
        The formatted file name.

    """

    fname = str(fname)  # converts other objects, e.g., pathlib.Path, to a string

    # normal file name: nothing to be done
    if "^" not in fname:
        return fname

    # pattern match
    parts = fname.split("^")
    sub = parts[1].split(".")
    coordstring = sub[0]
    m = re.match(r"_(\d+)_(\d+)(\D*)", coordstring)
    if m is not None:
        ix = int(m.group(1))
        iy = int(m.group(2))
        term = m.group(3)
    suffix = term + "." + ".".join(sub[1:])
    outname = "^".join(parts[:-1])
    outname = outname.format(ix, iy) + suffix
    return outname


def ReadFile(fname):
    """Wrapper to read a compressed file.

    This should read a file just like astropy.io.fits.open(fname, mode='readonly'),
    but works even if the file is compressed using compressutils.

    Parameters
    ----------
    fname : str or str-like
        File name to read.

    Notes
    -----
    This can also be used with the Python context manager, e.g.::

        with ReadFile('my.fits.gz') as f:
          ...

    File names with sepcific types of regular expressions are allowed, and unpacked by the ``_parser``
    function. Regular expressions are separated with the ^ character and contain a row and column index,
    followed by the suffix. For example::

        >>> _parser('hello_world/Q_02_31.fits') # no ^, regular file name
        'hello_world/Q_02_31.fits'
        >>> _parser('hello_world/Row{1:2d}/Q_{0:02d}_{1:02d}^_02_31.fits') # FITS file
        'hello_world/Row31/Q_02_31.fits'
        >>> _parser('hello_world/Row{1:2d}/Q_{0:02d}_{1:02d}^_02_12.fits.gz') # gzipped; suffix is copied over
        'hello_world/Row12/Q_02_12.fits.gz'

    """

    fname = _parser(fname)  # if the file name is a regular expression.

    _o = urlparse(fname)

    extraargs = {}
    match _o.scheme:
        case "" | "file":
            pass
        case "http" | "https":
            extraargs["use_fsspec"] = True
        case "s3":
            extraargs["use_fsspec"] = True
            extraargs["fsspec_kwargs"] = {"anon": True}
            # extraargs["fsspec_kwargs"] = {"key": "YOUR-SECRET-KEY-ID", "secret": "YOUR-SECRET-KEY"}
        case _:
            raise ValueError(f"Scheme {_o.scheme} not supported")

    _url = _o.geturl()

    f = fits.open(_url, **extraargs)

    if "CPRESS" not in [hdu.name for hdu in f]:
        return f
    else:
        f.close()

    # otherwise, make a decompressed version
    x = CompressedOutput(fname, extraargs=extraargs)
    x.decompress()
    return fits.HDUList(x.hdul)

if __name__ == "__main__":
    """Command line test.

   With 1 argument: makes compressed version of a file. .fits input only (no gz).

   With 5 arguments: main program arguments are::

     input_file out_file_compressed recovered recompressed ncycle

   ncycle=1 is sufficient to test functionality, but can do more than once to
   test for memory leaks.

   """

    if len(sys.argv) == 2:
        test1(sys.argv[1])

    if len(sys.argv) == 6:
        for _ in range(int(sys.argv[5])):
            test(sys.argv[:5])
