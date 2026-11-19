"""
Linh-An Tran
CSE 163
A small stand-in for the course's cse163_utils.assert_equals so the tests
can run anywhere. If you have the course's own cse163_utils.py, you can
replace this file with it; the tests only use assert_equals.
"""
import math

import pandas as pd

TOLERANCE = 0.001


def assert_equals(expected, received) -> None:
    """
    Checks that received equals expected and raises an AssertionError that
    shows both values if it does not. Floats are compared with a small
    tolerance, NaN equals NaN, lists / tuples / dicts are compared item by
    item, and pandas DataFrames and Series are compared with pandas'
    testing helpers (tolerant of tiny float differences).
    """
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(expected, received, check_dtype=False,
                                      atol=TOLERANCE)
    elif isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(expected, received, check_dtype=False,
                                       check_names=False, atol=TOLERANCE)
    elif isinstance(expected, float):
        both_nan = math.isnan(expected) and math.isnan(received)
        assert both_nan or math.isclose(expected, received,
                                        abs_tol=TOLERANCE), \
            f'Expected {expected} but received {received}'
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(received), \
            f'Expected {expected} but received {received}'
        for want, got in zip(expected, received):
            assert_equals(want, got)
    elif isinstance(expected, dict):
        assert expected.keys() == received.keys(), \
            f'Expected {expected} but received {received}'
        for key in expected:
            assert_equals(expected[key], received[key])
    else:
        assert expected == received, \
            f'Expected {expected} but received {received}'
