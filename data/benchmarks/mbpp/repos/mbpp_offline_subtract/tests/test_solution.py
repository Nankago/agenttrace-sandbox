import unittest

from solution import subtract


class GeneratedTests(unittest.TestCase):
    def test_case_1(self) -> None:
        assert subtract(5, 3) == 2

    def test_case_2(self) -> None:
        assert subtract(-1, -3) == 2
