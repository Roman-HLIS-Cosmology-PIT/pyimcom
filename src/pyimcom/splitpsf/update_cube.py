"""
Function to update cached files in a split-PSF iteration.
"""

import json
import os
import re
import shutil
import sys
from contextlib import suppress

from ..config import Config


def _colored(text, fg, bg):
    """
    Prints colored text.

    Format follows ANSI: 0=black, 1=red, 2=green, 3=yellow, 4=blue, 5=magenta, 6=cyan, 7=white, 9=default.

    Parameters
    ----------
    text : str
        The text to print.
    fg : int
        Foreground color.
    bg : int
        Background color.

    Returns
    -------
    str
        Formatted string with ANSI codes.

    """

    fgcolor = "\033[3" + f"{fg:1d}" + "m"
    bgcolor = "\033[4" + f"{bg:1d}" + "m"
    reset = "\033[0m"
    return f"{bgcolor}{fgcolor}{text}{reset}"


def update(cfg_file, proceed=True):
    """
    Main function to update the data cubes to the next iteration.

    Parameters
    ----------
    cfg_file : str or str-like
        The configuration file.
    proceed : bool, optional
        Whether to actually move the files (only turn off for testing).

    Returns
    -------
    None

    """

    cfgdata = Config(cfg_file)

    # separate the path from the inlayercache info
    m = re.search(r"^(.*)\/(.*)", cfgdata.inlayercache)
    if m:
        path = m.group(1)
        exp = m.group(2)

    # create empty list of exposures
    idsca = []

    # find all the fits files and add them to the list
    print("Searching for files: " + cfgdata.inlayercache + "********_**.fits")
    for _, _, files in os.walk(path):
        for file in files:
            if file.startswith(exp):
                m = re.search(r"_(\d{8})_(\d{2})\.fits$", file[len(exp) :])
                if m:
                    idsca.append((int(m.group(1)), int(m.group(2))))

    # get iteration number
    iter = 0
    iterfile = cfgdata.inlayercache + "_iter.txt"
    oldcfgfile = cfgdata.inlayercache + "_oldcfg.json"
    if os.path.exists(iterfile):
        with open(iterfile, "r") as f:
            iter = int(f.read().split()[0])
    if iter == 0:
        with suppress(FileNotFoundError):
            os.remove(oldcfgfile)

    # Check that all the files are there!
    all_files = True
    orig_file = []
    sub_file = []
    target_file = []
    for id, sca in idsca:
        orig_file.append(cfgdata.inlayercache + f"_{id:08d}_{sca:02d}.fits")
        sub_file.append(cfgdata.inlayercache + f"_{id:08d}_{sca:02d}_subI.fits")
        target_file.append(cfgdata.inlayercache + f"_{id:08d}_{sca:02d}_{iter:02d}iter.fits")
        ex = os.path.exists(orig_file[-1]) and os.path.exists(sub_file[-1])
        if not ex:
            all_files = False
    if all_files:
        print("Checking if all files present: " + _colored("Passed", 7, 0))
    else:
        print("Checking if all files present: " + _colored("Failed", 7, 1))

    if all_files:
        # move files
        N = len(sub_file)
        print(f"Moving {N} files ...")
        for j in range(N):
            print(f"    {j:3d}/{N:3d}: {sub_file[j]} --> {orig_file[j]} --> {target_file[j]}")
            if proceed:
                shutil.move(orig_file[j], target_file[j])
                shutil.move(sub_file[j], orig_file[j])

        # update previous configuration
        prev_cfgs = {}
        if os.path.exists(oldcfgfile):
            with open(oldcfgfile) as f:
                prev_cfgs = json.load(f)
            os.remove(oldcfgfile)
        prev_cfgs[f"CONFIG{iter:d}"] = json.loads(cfgdata.to_file(None))
        with open(oldcfgfile, "w") as f:
            f.write(json.dumps(prev_cfgs))

        # update iteration counter
        iter += 1
        with open(iterfile, "w") as f:
            f.write(f"{iter:d}")


# Command line driver
if __name__ == "__main__":
    proceed = True
    if len(sys.argv) > 2:
        proceed = False
    print(f">> proceed = {proceed}")
    update(sys.argv[1], proceed=proceed)
