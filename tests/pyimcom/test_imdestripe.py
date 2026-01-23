"""Test functions for imdestripe."""

from pyimcom import imdestripe


def test_get_ids():
    """Test function for splitting an obsid,sca pair."""

    s = "670_16"  # string to parse
    obsid, scaid = imdestripe.get_ids(s)
    print(obsid, scaid)
    # check if we parsed correctly
    assert obsid == 670
    assert scaid == 16
