import unittest

from solution import square


class GeneratedTests(unittest.TestCase):
    def test_case_1(self) -> None:
        assert square(4) == 16

    def test_case_2(self) -> None:
        assert square(-3) == 9
