import unittest
import numpy as np

import algs.utils as utils


class TestPatternIterator(unittest.TestCase):
    """Tests the PatternIterator class."""

    def test_PatternIterator(self):
        # no forbidden bits
        it = utils.PatternIterator(np.array([False, False, False]))
        self.assertEqual(len(it), 3)

        self.assertTrue(it.has_next())
        np.testing.assert_allclose(it.get(), [0, 0, 0])
        self.assertEqual(it.get_int(), 0)

        np.testing.assert_allclose(it.next(), [1, 0, 0])
        self.assertTrue(it.has_next())
        np.testing.assert_allclose(it.get(), [1, 0, 0])
        self.assertEqual(it.get_int(), 1)

        np.testing.assert_allclose(it.next(), [0, 1, 0])
        self.assertTrue(it.has_next())
        np.testing.assert_allclose(it.get(), [0, 1, 0])
        self.assertEqual(it.get_int(), 2)

        np.testing.assert_allclose(it.next(), [1, 1, 0])
        self.assertTrue(it.has_next())
        np.testing.assert_allclose(it.get(), [1, 1, 0])
        self.assertEqual(it.get_int(), 3)

        np.testing.assert_allclose(it.next(), [0, 0, 1])
        self.assertTrue(it.has_next())
        np.testing.assert_allclose(it.get(), [0, 0, 1])
        self.assertEqual(it.get_int(), 4)

        np.testing.assert_allclose(it.next(), [1, 0, 1])
        self.assertTrue(it.has_next())
        np.testing.assert_allclose(it.get(), [1, 0, 1])
        self.assertEqual(it.get_int(), 5)

        np.testing.assert_allclose(it.next(), [0, 1, 1])
        self.assertTrue(it.has_next())
        np.testing.assert_allclose(it.get(), [0, 1, 1])
        self.assertEqual(it.get_int(), 6)

        np.testing.assert_allclose(it.next(), [1, 1, 1])
        self.assertFalse(it.has_next())
        np.testing.assert_allclose(it.get(), [1, 1, 1])
        self.assertEqual(it.get_int(), 7)

        self.assertIsNone(it.next())
        self.assertFalse(it.has_next())
        np.testing.assert_allclose(it.get(), [1, 1, 1])
        self.assertEqual(it.get_int(), 7)

        self.assertIsNone(it.next())  # should be no side effects
        self.assertFalse(it.has_next())
        np.testing.assert_allclose(it.get(), [1, 1, 1])
        self.assertEqual(it.get_int(), 7)

        # some forbidden bits
        it = utils.PatternIterator(np.array([False, True, False]))

        self.assertTrue(it.has_next())
        np.testing.assert_allclose(it.get(), [0, 0, 0])
        self.assertEqual(it.get_int(), 0)

        np.testing.assert_allclose(it.next(), [1, 0, 0])
        self.assertTrue(it.has_next())
        np.testing.assert_allclose(it.get(), [1, 0, 0])
        self.assertEqual(it.get_int(), 1)

        np.testing.assert_allclose(it.next(), [0, 0, 1])
        self.assertTrue(it.has_next())
        np.testing.assert_allclose(it.get(), [0, 0, 1])
        self.assertEqual(it.get_int(), 4)

        np.testing.assert_allclose(it.next(), [1, 0, 1])
        self.assertFalse(it.has_next())
        np.testing.assert_allclose(it.get(), [1, 0, 1])
        self.assertEqual(it.get_int(), 5)

        self.assertIsNone(it.next())
        self.assertFalse(it.has_next())
        np.testing.assert_allclose(it.get(), [1, 0, 1])
        self.assertEqual(it.get_int(), 5)

        self.assertIsNone(it.next())  # should be no side effects
        self.assertFalse(it.has_next())
        np.testing.assert_allclose(it.get(), [1, 0, 1])
        self.assertEqual(it.get_int(), 5)

        it = utils.PatternIterator(np.array([False, True, True]))

        self.assertTrue(it.has_next())
        np.testing.assert_allclose(it.get(), [0, 0, 0])
        self.assertEqual(it.get_int(), 0)

        np.testing.assert_allclose(it.next(), [1, 0, 0])
        self.assertFalse(it.has_next())
        np.testing.assert_allclose(it.get(), [1, 0, 0])
        self.assertEqual(it.get_int(), 1)

        self.assertIsNone(it.next())
        self.assertFalse(it.has_next())
        np.testing.assert_allclose(it.get(), [1, 0, 0])
        self.assertEqual(it.get_int(), 1)

        self.assertIsNone(it.next())  # should be no side effects
        self.assertFalse(it.has_next())
        np.testing.assert_allclose(it.get(), [1, 0, 0])
        self.assertEqual(it.get_int(), 1)

        it = utils.PatternIterator(np.array([True, True, False]))

        self.assertTrue(it.has_next())
        np.testing.assert_allclose(it.get(), [0, 0, 0])
        self.assertEqual(it.get_int(), 0)

        np.testing.assert_allclose(it.next(), [0, 0, 1])
        self.assertFalse(it.has_next())
        np.testing.assert_allclose(it.get(), [0, 0, 1])
        self.assertEqual(it.get_int(), 4)

        self.assertIsNone(it.next())
        self.assertFalse(it.has_next())
        np.testing.assert_allclose(it.get(), [0, 0, 1])
        self.assertEqual(it.get_int(), 4)

        self.assertIsNone(it.next())  # should be no side effects
        self.assertFalse(it.has_next())
        np.testing.assert_allclose(it.get(), [0, 0, 1])

        it = utils.PatternIterator(np.array([True, False, True]))

        self.assertTrue(it.has_next())
        np.testing.assert_allclose(it.get(), [0, 0, 0])
        self.assertEqual(it.get_int(), 0)

        np.testing.assert_allclose(it.next(), [0, 1, 0])
        self.assertFalse(it.has_next())
        np.testing.assert_allclose(it.get(), [0, 1, 0])
        self.assertEqual(it.get_int(), 2)

        self.assertIsNone(it.next())
        self.assertFalse(it.has_next())
        np.testing.assert_allclose(it.get(), [0, 1, 0])
        self.assertEqual(it.get_int(), 2)

        self.assertIsNone(it.next())  # should be no side effects
        self.assertFalse(it.has_next())
        np.testing.assert_allclose(it.get(), [0, 1, 0])
        self.assertEqual(it.get_int(), 2)

        it = utils.PatternIterator(np.array([True, True, True]))

        self.assertFalse(it.has_next())
        np.testing.assert_allclose(it.get(), [0, 0, 0])
        self.assertEqual(it.get_int(), 0)

        self.assertIsNone(it.next())
        self.assertFalse(it.has_next())
        np.testing.assert_allclose(it.get(), [0, 0, 0])
        self.assertEqual(it.get_int(), 0)

        self.assertIsNone(it.next())  # should be no side effects
        self.assertFalse(it.has_next())
        np.testing.assert_allclose(it.get(), [0, 0, 0])
        self.assertEqual(it.get_int(), 0)

        # handling large integers
        it = utils.PatternIterator(np.array([False] + [True] * 80 + [False]))

        self.assertTrue(it.has_next())
        np.testing.assert_allclose(it.get(), [0] * 82)
        self.assertEqual(it.get_int(), 0)

        np.testing.assert_allclose(it.next(), [1] + [0] * 81)
        self.assertTrue(it.has_next())
        np.testing.assert_allclose(it.get(), [1] + [0] * 81)
        self.assertEqual(it.get_int(), 1)

        print(it._int_val, it._cursor, it._avail[it._cursor])
        np.testing.assert_allclose(it.next(), [0] * 81 + [1])
        print(it._int_val, it._cursor, it._avail[it._cursor])
        self.assertTrue(it.has_next())
        np.testing.assert_allclose(it.get(), [0] * 81 + [1])
        self.assertEqual(it.get_int(), 2417851639229258349412352)

        np.testing.assert_allclose(it.next(), [1] + [0] * 80 + [1])
        self.assertFalse(it.has_next())
        np.testing.assert_allclose(it.get(), [1] + [0] * 80 + [1])
        self.assertEqual(it.get_int(), 2417851639229258349412353)

        self.assertIsNone(it.next())
        self.assertFalse(it.has_next())
        np.testing.assert_allclose(it.get(), [1] + [0] * 80 + [1])
        self.assertEqual(it.get_int(), 2417851639229258349412353)

        self.assertIsNone(it.next())  # should be no side effects
        self.assertFalse(it.has_next())
        np.testing.assert_allclose(it.get(), [1] + [0] * 80 + [1])
        self.assertEqual(it.get_int(), 2417851639229258349412353)
