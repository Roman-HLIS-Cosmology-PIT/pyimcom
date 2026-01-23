"""Test functions for imdestripe."""

from pyimcom import imdestripe


def test_get_ids():
    """Test function for splitting an obsid,sca pair."""

    s = "mybeginning_670_16.myending"  # string to parse
    obsid, scaid = imdestripe.get_ids(s)
    print(obsid, scaid)
    # check if we parsed correctly
    assert obsid == "670"
    assert scaid == "16"
