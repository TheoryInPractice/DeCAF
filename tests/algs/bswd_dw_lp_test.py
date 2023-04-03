import unittest
import numpy as np
import networkx as nx

import algs.bswd_dw_lp as bswd_dw_lp
import algs.utils_misc as utils_misc

# small instances for testing
instances = [
    # ------------------------------------------------------------
    #    0: n=4, k=3
    # ------------------------------------------------------------
    np.array([
        [np.inf, 2, 2, 2],
        [2, np.inf, 1, 1],
        [2, 1, np.inf, 1],
        [2, 1, 1, np.inf],
    ]),
    # ------------------------------------------------------------
    #    1: n=4, k=3
    # ------------------------------------------------------------
    np.array([
        [np.inf, 4, 5, 3],
        [4, np.inf, 3, 1],
        [5, 3, np.inf, 2],
        [3, 1, 2, np.inf],
    ]),
    # ------------------------------------------------------------
    #    2: n=4, k=4
    # ------------------------------------------------------------
    np.array([
        [np.inf, 8, 6, 5],
        [8, np.inf, 2, 2],
        [6, 2, np.inf, 2],
        [5, 2, 2, np.inf],
    ]),
    # ------------------------------------------------------------
    #    3: n=4, k=3
    # ------------------------------------------------------------
    np.array([
        [5, 2, 5, 3],
        [2, 2, 2, 0],
        [5, 2, 9, 3],
        [3, 0, 3, 3],
    ]),
    # ------------------------------------------------------------
    #    4: n=5, k=3
    # ------------------------------------------------------------
    np.array([
        [np.inf, 3, 3, 0, 0],
        [3, np.inf, 3, 4, 4],
        [3, 3, np.inf, 4, 4],
        [0, 4, 4, np.inf, 8],
        [0, 4, 4, 8, np.inf],
    ]),
    # ------------------------------------------------------------
    #    5: n=5, k=5
    # ------------------------------------------------------------
    np.array([
        [np.inf, 3, 3, 0, 0],
        [3, 9, 3, 4, 4],
        [3, 3, np.inf, 4, 4],
        [0, 4, 4, 10, 8],
        [0, 4, 4, 8, 8],
    ]),
    # ------------------------------------------------------------
    #    6: n=5, k=4
    # -----------------------------------------------------------
    np.array([
        [np.inf, 3, 3, 0, 0],
        [3, 9, 3, 6, 4],
        [3, 3, np.inf, 4, 4],
        [0, 6, 4, 10, 8],
        [0, 4, 4, 8, 8],
    ]),
    # ------------------------------------------------------------
    #    7: n=5, k=1
    # -----------------------------------------------------------
    np.array([
        [np.inf, 2, 2, 2, 2],
        [2, np.inf, 2, 2, 2],
        [2, 2, np.inf, 2, 2],
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, np.inf],
    ]),
    # ------------------------------------------------------------
    #    8: n=4, k=4
    # -----------------------------------------------------------
    np.array([
        [15, 1, 2, 3],
        [1, np.inf, 0, 0],
        [2, 0, np.inf, 0],
        [3, 0, 0, np.inf],
    ]),
    # ------------------------------------------------------------
    #    9: n=3, k=6
    # -----------------------------------------------------------
    np.array([
        [18, 11, 13],
        [11, 18, 12],
        [13, 12, 21],
    ]),
    # ------------------------------------------------------------
    #    10: n=15, k=3
    # -----------------------------------------------------------
    np.array([
        [np.inf, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, np.inf, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, np.inf, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 2, np.inf, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 2, 2, np.inf, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, np.inf, 3, 3, 3, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, np.inf, 3, 3, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 3, np.inf, 3, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 3, 3, np.inf, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 3, 3, 3, np.inf, 4, 4, 4, 4, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, np.inf, 4, 4, 4, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, np.inf, 4, 4, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, np.inf, 4, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, np.inf, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, np.inf],
    ]),
]

# expected results
# ------------------------------------------------------------
#    0
# ------------------------------------------------------------
expected_0_b = [
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 0],
]
expected_0_w = [1, 1, 1]

# ------------------------------------------------------------
#    1
# ------------------------------------------------------------
expected_1_b = [
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 0],
]
expected_1_w = [1, 2, 3]

# ------------------------------------------------------------
#    2
# ------------------------------------------------------------
expected_2_b = [
    [0, 1, 1, 1],
    [1, 0, 1, 0],
    [1, 0, 0, 1],
    [1, 1, 0, 0],
]
expected_2_w = [2, 5, 8, 6]

# ------------------------------------------------------------
#    3
# ------------------------------------------------------------
expected_3_b = [
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
    [0, 1, 0],
]
expected_3_w = [4, 3, 2]

# ------------------------------------------------------------
#    4
# ------------------------------------------------------------
expected_4_b = [
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 0],
]
expected_4_w = [4, 4, 3]

# ------------------------------------------------------------
#    5
# ------------------------------------------------------------
expected_5_b = [
    [0, 0, 0, 0, 1],
    [0, 0, 1, 1, 1],
    [0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
]
expected_5_w = [2, 4, 2, 4, 3]

# ------------------------------------------------------------
#    6
# ------------------------------------------------------------
expected_6_b = [
    [0, 0, 0, 1],
    [0, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 0],
    [1, 0, 1, 0],
]
expected_6_w = [4, 2, 4, 3]

# ------------------------------------------------------------
#    7
# ------------------------------------------------------------
expected_7_b = [
    [1],
    [1],
    [1],
    [1],
    [1],
]
expected_7_w = [2]

# ------------------------------------------------------------
#    8
# ------------------------------------------------------------
expected_8_b = [
    [1, 1, 1, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 1, 0, 0],
]
expected_8_w = [9, 3, 1, 2]

# ------------------------------------------------------------
#    9
# ------------------------------------------------------------
expected_9_b = [
    [0, 0, 0, 1, 1, 1],
    [0, 1, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 1],
]
expected_9_w = [7, 6, 1, 5, 2, 11]

# ------------------------------------------------------------
#    10
# ------------------------------------------------------------
expected_10_b = [
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 1],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [1, 1, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
]
expected_10_w = [4, 3, 2]


class TestBswdDwLp(unittest.TestCase):
    """Tests the bswd_dw_lp module."""

    def to_diagonal(self, xs):
        """
        Converts an array to the diagonal matrix.
        """
        n = len(xs)
        return [[xs[i] if i == j else 0 for i in range(n)] for j in range(n)]

    def check_decomposition(self, actual_b, actual_w, expected_b, expected_w):
        w = [actual_w[i][i] for i in range(actual_w.shape[0])]
        np.testing.assert_allclose(actual_w, self.to_diagonal(w))
        actual = set((tuple(round(x) for x in xs), y) for xs, y in zip(np.array(actual_b).transpose(), w))
        expected = set((tuple(round(x) for x in xs), y) for xs, y in zip(np.array(expected_b).transpose(), expected_w))
        self.assertSetEqual(actual, expected)

    def test_BSWD_DW(self):
        """Tests BSWD_DW()."""

        v2 = False
        v3 = False

        # instance 0
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[0], 2, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])  # infeasible
        np.testing.assert_allclose(W, [[-1]])  # infeasible
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[0], 3, v2, v3, None)
        np.testing.assert_allclose(B, expected_0_b)
        np.testing.assert_allclose(W, self.to_diagonal(expected_0_w))
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[0], 4, v2, v3, None)
        np.testing.assert_allclose(B, [[0] + s for s in expected_0_b])
        np.testing.assert_allclose(W, self.to_diagonal([0] + expected_0_w))
        self.assertGreater(num_lp_runs, 0)

        # instance 1
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[1], 2, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[1], 3, v2, v3, None)
        np.testing.assert_allclose(B, expected_1_b)
        np.testing.assert_allclose(W, self.to_diagonal(expected_1_w))
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[1], 4, v2, v3, None)
        np.testing.assert_allclose(B, [[0] + s for s in expected_1_b])
        np.testing.assert_allclose(W, self.to_diagonal([0] + expected_1_w))
        self.assertGreater(num_lp_runs, 0)

        # instance 2
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[2], 3, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[2], 4, v2, v3, None)
        np.testing.assert_allclose(B, expected_2_b)
        np.testing.assert_allclose(W, self.to_diagonal(expected_2_w))
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[2], 5, v2, v3, None)
        np.testing.assert_allclose(B, [[0] + s for s in expected_2_b])
        np.testing.assert_allclose(W, self.to_diagonal([0] + expected_2_w))
        self.assertGreater(num_lp_runs, 0)

        # instance 3
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[3], 2, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[3], 3, v2, v3, None)
        np.testing.assert_allclose(B, expected_3_b)
        np.testing.assert_allclose(W, self.to_diagonal(expected_3_w))
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[3], 4, v2, v3, None)
        np.testing.assert_allclose(B, [[0] + s for s in expected_3_b])
        np.testing.assert_allclose(W, self.to_diagonal([0] + expected_3_w))
        self.assertGreater(num_lp_runs, 0)

        # instance 4
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[4], 2, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[4], 3, v2, v3, None)

        np.testing.assert_allclose(B, expected_4_b)
        np.testing.assert_allclose(W, self.to_diagonal(expected_4_w))
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[4], 4, v2, v3, None)
        np.testing.assert_allclose(B, [[0] + s for s in expected_4_b])
        np.testing.assert_allclose(W, self.to_diagonal([0] + expected_4_w))
        self.assertGreater(num_lp_runs, 0)

        # instance 5
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[5], 4, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[5], 5, v2, v3, None)
        np.testing.assert_allclose(B, expected_5_b)
        np.testing.assert_allclose(W, self.to_diagonal(expected_5_w))
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[5], 6, v2, v3, None)
        np.testing.assert_allclose(B, [[0] + s for s in expected_5_b])
        np.testing.assert_allclose(W, self.to_diagonal([0] + expected_5_w))
        self.assertGreater(num_lp_runs, 0)

        # instance 6
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[6], 3, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[6], 4, v2, v3, None)
        np.testing.assert_allclose(B, expected_6_b)
        np.testing.assert_allclose(W, self.to_diagonal(expected_6_w))
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[6], 5, v2, v3, None)
        np.testing.assert_allclose(B, [[0] + s for s in expected_6_b])
        np.testing.assert_allclose(W, self.to_diagonal([0] + expected_6_w))
        self.assertGreater(num_lp_runs, 0)

        # instance 7
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[7], 0, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertEqual(num_lp_runs, 0)  # no LP runs

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[7], 1, v2, v3, None)
        np.testing.assert_allclose(B, expected_7_b)
        np.testing.assert_allclose(W, self.to_diagonal(expected_7_w))
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[7], 2, v2, v3, None)
        np.testing.assert_allclose(B, [[0] + s for s in expected_7_b])
        np.testing.assert_allclose(W, self.to_diagonal([0] + expected_7_w))
        self.assertGreater(num_lp_runs, 0)

        # instance 8
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[8], 3, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[8], 4, v2, v3, None)
        np.testing.assert_allclose(B, expected_8_b)
        np.testing.assert_allclose(W, self.to_diagonal(expected_8_w))
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[8], 5, v2, v3, None)
        np.testing.assert_allclose(B, [[0] + s for s in expected_8_b])
        np.testing.assert_allclose(W, self.to_diagonal([0] + expected_8_w))
        self.assertGreater(num_lp_runs, 0)

        # instance 9
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[9], 5, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[9], 6, v2, v3, None)
        np.testing.assert_allclose(B, expected_9_b)
        np.testing.assert_allclose(W, self.to_diagonal(expected_9_w))
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[9], 7, v2, v3, None)
        np.testing.assert_allclose(B, [[0] + s for s in expected_9_b])
        np.testing.assert_allclose(W, self.to_diagonal([0] + expected_9_w))
        self.assertGreater(num_lp_runs, 0)

        # instance 10
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[10], 2, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[10], 3, v2, v3, None)
        np.testing.assert_allclose(B, expected_10_b)
        np.testing.assert_allclose(W, self.to_diagonal(expected_10_w))
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[10], 4, v2, v3, None)
        np.testing.assert_allclose(B, [[0] + s for s in expected_10_b])
        np.testing.assert_allclose(W, self.to_diagonal([0] + expected_10_w))
        self.assertGreater(num_lp_runs, 0)

    def test_BSWD_DW_v2(self):
        """Tests BSWD_DW() with version 2 features."""

        v2 = True
        v3 = False

        # instance 0
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[0], 2, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])  # infeasible
        np.testing.assert_allclose(W, [[-1]])  # infeasible
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[0], 3, v2, v3, None)
        self.check_decomposition(B, W, expected_0_b, expected_0_w)
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[0], 4, v2, v3, None)
        self.check_decomposition(B, W, [[0] + s for s in expected_0_b], [0] + expected_0_w)
        self.assertGreater(num_lp_runs, 0)

        # instance 1
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[1], 2, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[1], 3, v2, v3, None)
        self.check_decomposition(B, W, expected_1_b, expected_1_w)
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[1], 4, v2, v3, None)
        self.check_decomposition(B, W, [[0] + s for s in expected_1_b], [0] + expected_1_w)
        self.assertGreater(num_lp_runs, 0)

        # instance 2
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[2], 3, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[2], 4, v2, v3, None)
        self.check_decomposition(B, W, expected_2_b, expected_2_w)
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[2], 5, v2, v3, None)
        self.check_decomposition(B, W, [[0] + s for s in expected_2_b], [0] + expected_2_w)
        self.assertGreater(num_lp_runs, 0)

        # instance 3
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[3], 2, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[3], 3, v2, v3, None)
        self.check_decomposition(B, W, expected_3_b, expected_3_w)
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[3], 4, v2, v3, None)
        self.check_decomposition(B, W, [[0] + s for s in expected_3_b], [0] + expected_3_w)
        self.assertGreater(num_lp_runs, 0)

        # instance 4
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[4], 2, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[4], 3, v2, v3, None)
        self.check_decomposition(B, W, expected_4_b, expected_4_w)
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[4], 4, v2, v3, None)
        self.check_decomposition(B, W, [[0] + s for s in expected_4_b], [0] + expected_4_w)
        self.assertGreater(num_lp_runs, 0)

        # instance 5
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[5], 4, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[5], 5, v2, v3, None)
        self.check_decomposition(B, W, expected_5_b, expected_5_w)
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[5], 6, v2, v3, None)
        self.check_decomposition(B, W, [[0] + s for s in expected_5_b], [0] + expected_5_w)
        self.assertGreater(num_lp_runs, 0)

        # instance 6
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[6], 3, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[6], 4, v2, v3, None)
        self.check_decomposition(B, W, expected_6_b, expected_6_w)
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[6], 5, v2, v3, None)
        self.check_decomposition(B, W, [[0] + s for s in expected_6_b], [0] + expected_6_w)
        self.assertGreater(num_lp_runs, 0)

        # instance 7
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[7], 0, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertEqual(num_lp_runs, 0)  # no LP runs

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[7], 1, v2, v3, None)
        self.check_decomposition(B, W, expected_7_b, expected_7_w)
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[7], 2, v2, v3, None)
        self.check_decomposition(B, W, [[0] + s for s in expected_7_b], [0] + expected_7_w)
        self.assertGreater(num_lp_runs, 0)

        # instance 8
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[8], 3, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[8], 4, v2, v3, None)
        self.check_decomposition(B, W, expected_8_b, expected_8_w)
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[8], 5, v2, v3, None)
        self.check_decomposition(B, W, [[0] + s for s in expected_8_b], [0] + expected_8_w)
        self.assertGreater(num_lp_runs, 0)

        # instance 9
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[9], 5, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[9], 6, v2, v3, None)
        self.check_decomposition(B, W, expected_9_b, expected_9_w)
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[9], 7, v2, v3, None)
        self.check_decomposition(B, W, [[0] + s for s in expected_9_b], [0] + expected_9_w)
        self.assertGreater(num_lp_runs, 0)

        # instance 10
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[10], 2, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[10], 3, v2, v3, None)
        self.check_decomposition(B, W, expected_10_b, expected_10_w)
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[10], 4, v2, v3, None)
        self.check_decomposition(B, W, [[0] + s for s in expected_10_b], [0] + expected_10_w)
        self.assertGreater(num_lp_runs, 0)

    def test_BSWD_DW_v3(self):
        """Tests BSWD_DW() with version 3 features."""

        v2 = False
        v3 = True

        # instance 0
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[0], 2, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])  # infeasible
        np.testing.assert_allclose(W, [[-1]])  # infeasible
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[0], 3, v2, v3, None)
        self.check_decomposition(B, W, expected_0_b, expected_0_w)
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[0], 4, v2, v3, None)
        self.check_decomposition(B, W, [[0] + s for s in expected_0_b], [0] + expected_0_w)
        self.assertGreater(num_lp_runs, 0)

        # instance 1
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[1], 2, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[1], 3, v2, v3, None)
        self.check_decomposition(B, W, expected_1_b, expected_1_w)
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[1], 4, v2, v3, None)
        self.check_decomposition(B, W, [[0] + s for s in expected_1_b], [0] + expected_1_w)
        self.assertGreater(num_lp_runs, 0)

        # instance 2
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[2], 3, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[2], 4, v2, v3, None)
        self.check_decomposition(B, W, expected_2_b, expected_2_w)
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[2], 5, v2, v3, None)
        self.check_decomposition(B, W, [[0] + s for s in expected_2_b], [0] + expected_2_w)
        self.assertGreater(num_lp_runs, 0)

        # instance 3
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[3], 2, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[3], 3, v2, v3, None)
        self.check_decomposition(B, W, expected_3_b, expected_3_w)
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[3], 4, v2, v3, None)
        self.check_decomposition(B, W, [[0] + s for s in expected_3_b], [0] + expected_3_w)
        self.assertGreater(num_lp_runs, 0)

        # instance 4
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[4], 2, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[4], 3, v2, v3, None)
        self.check_decomposition(B, W, expected_4_b, expected_4_w)
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[4], 4, v2, v3, None)
        self.check_decomposition(B, W, [[0] + s for s in expected_4_b], [0] + expected_4_w)
        self.assertGreater(num_lp_runs, 0)

        # instance 5
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[5], 4, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[5], 5, v2, v3, None)
        self.check_decomposition(B, W, expected_5_b, expected_5_w)
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[5], 6, v2, v3, None)
        self.check_decomposition(B, W, [[0] + s for s in expected_5_b], [0] + expected_5_w)
        self.assertGreater(num_lp_runs, 0)

        # instance 6
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[6], 3, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[6], 4, v2, v3, None)
        self.check_decomposition(B, W, expected_6_b, expected_6_w)
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[6], 5, v2, v3, None)
        self.check_decomposition(B, W, [[0] + s for s in expected_6_b], [0] + expected_6_w)
        self.assertGreater(num_lp_runs, 0)

        # instance 7
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[7], 0, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertEqual(num_lp_runs, 0)  # no LP runs

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[7], 1, v2, v3, None)
        self.check_decomposition(B, W, expected_7_b, expected_7_w)
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[7], 2, v2, v3, None)
        self.check_decomposition(B, W, [[0] + s for s in expected_7_b], [0] + expected_7_w)
        self.assertGreater(num_lp_runs, 0)

        # instance 8
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[8], 3, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[8], 4, v2, v3, None)
        self.check_decomposition(B, W, expected_8_b, expected_8_w)
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[8], 5, v2, v3, None)
        self.check_decomposition(B, W, [[0] + s for s in expected_8_b], [0] + expected_8_w)
        self.assertGreater(num_lp_runs, 0)

        # instance 9
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[9], 5, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[9], 6, v2, v3, None)
        self.check_decomposition(B, W, expected_9_b, expected_9_w)
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[9], 7, v2, v3, None)
        self.check_decomposition(B, W, [[0] + s for s in expected_9_b], [0] + expected_9_w)
        self.assertGreater(num_lp_runs, 0)

        # instance 10
        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[10], 2, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[10], 3, v2, v3, None)
        self.check_decomposition(B, W, expected_10_b, expected_10_w)
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(instances[10], 4, v2, v3, None)
        self.check_decomposition(B, W, [[0] + s for s in expected_10_b], [0] + expected_10_w)
        self.assertGreater(num_lp_runs, 0)

    def test_BSWD_DW_v2_large(self):
        """Tests BSWD_DW() with version 2 features with larger instances"""

        v2 = True
        v3 = False
        # ------------------------------------------------------------
        #    1
        # ------------------------------------------------------------
        G1 = nx.read_weighted_edgelist('tests/resources/g_4_n428.txt', nodetype=int)
        A1 = utils_misc.get_wildcard_adjacency(G1).to_numpy()
        expected_b = np.loadtxt('tests/resources/g_4_n248_b.csv', delimiter=',')
        expected_w = np.loadtxt('tests/resources/g_4_n248_w.csv', delimiter=',')

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(A1, 4, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(A1, 5, v2, v3, None)
        np.testing.assert_allclose(B, expected_b)
        np.testing.assert_allclose(W, expected_w)
        self.assertGreater(num_lp_runs, 0)

        # ------------------------------------------------------------
        #    2
        # ------------------------------------------------------------
        G2 = nx.read_weighted_edgelist('tests/resources/g_5_n271.txt', nodetype=int)
        A2 = utils_misc.get_wildcard_adjacency(G2).to_numpy()
        expected_b = np.loadtxt('tests/resources/g_5_n271_b.csv', delimiter=',')
        expected_w = np.loadtxt('tests/resources/g_5_n271_w.csv', delimiter=',')

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(A2, 4, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(A2, 6, v2, v3, None)
        np.testing.assert_allclose(B, expected_b)
        np.testing.assert_allclose(W, expected_w)
        self.assertGreater(num_lp_runs, 0)

    def test_BSWD_DW_v3_large(self):
        """Tests BSWD_DW() with version 2 features with larger instances"""

        v2 = False
        v3 = True
        # ------------------------------------------------------------
        #    1
        # ------------------------------------------------------------
        G1 = nx.read_weighted_edgelist('tests/resources/g_4_n428.txt', nodetype=int)
        A1 = utils_misc.get_wildcard_adjacency(G1).to_numpy()
        expected_b = np.loadtxt('tests/resources/g_4_n248_b.csv', delimiter=',')
        expected_w = np.loadtxt('tests/resources/g_4_n248_w.csv', delimiter=',')

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(A1, 4, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(A1, 5, v2, v3, None)
        np.testing.assert_allclose(B, expected_b)
        np.testing.assert_allclose(W, expected_w)
        self.assertGreater(num_lp_runs, 0)

        # ------------------------------------------------------------
        #    2
        # ------------------------------------------------------------
        G2 = nx.read_weighted_edgelist('tests/resources/g_5_n271.txt', nodetype=int)
        A2 = utils_misc.get_wildcard_adjacency(G2).to_numpy()
        expected_b = np.loadtxt('tests/resources/g_5_n271_b.csv', delimiter=',')
        expected_w = np.loadtxt('tests/resources/g_5_n271_w.csv', delimiter=',')

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(A2, 4, v2, v3, None)
        np.testing.assert_allclose(B, [[-1]])
        np.testing.assert_allclose(W, [[-1]])
        self.assertGreater(num_lp_runs, 0)

        B, W, num_lp_runs = bswd_dw_lp.BSWD_DW(A2, 6, v2, v3, None)
        np.testing.assert_allclose(B, expected_b)
        np.testing.assert_allclose(W, expected_w)
        self.assertGreater(num_lp_runs, 0)


class TestSignatureManager(unittest.TestCase):
    """Tests the SignatureManager class."""

    def test_SignatureManager(self):
        """Tests SignatureManager methods."""
        block_ids = [0, -1, 0, 1, 2, 1, 0, 2, 1, -1, 1, -1]
        sig_mng = bswd_dw_lp.SignatureManager(block_ids)

        self.assertTrue(sig_mng.is_valid(0, 3))
        sig_mng.add_signature(0, 3)

        # a different block may not have the same signature
        self.assertFalse(sig_mng.is_valid(1, 3))
        self.assertFalse(sig_mng.is_valid(3, 3))
        self.assertFalse(sig_mng.is_valid(4, 3))
        self.assertFalse(sig_mng.is_valid(5, 3))

        # the same block may have the same signature
        self.assertTrue(sig_mng.is_valid(2, 3))
        self.assertTrue(sig_mng.is_valid(6, 3))

        sig_mng.add_signature(2, 3)

        # now block 0 becomes identical
        self.assertTrue(sig_mng.is_valid(6, 3))
        self.assertFalse(sig_mng.is_valid(6, 1))
        self.assertFalse(sig_mng.is_valid(6, 7))
        self.assertFalse(sig_mng.is_valid(6, 5))
        sig_mng.add_signature(6, 3)

        # reset some signatures
        sig_mng.remove_signature(6, 3)
        sig_mng.remove_signature(2, 3)

        # the same block may not have subsets/supersets
        self.assertFalse(sig_mng.is_valid(2, 1))
        self.assertFalse(sig_mng.is_valid(6, 7))
        self.assertTrue(sig_mng.is_valid(6, 5))
        sig_mng.add_signature(6, 5)

        # now block 0 cannot add identical signatures
        self.assertFalse(sig_mng.is_valid(2, 3))
        self.assertFalse(sig_mng.is_valid(2, 5))
        self.assertFalse(sig_mng.is_valid(2, 7))
        self.assertFalse(sig_mng.is_valid(2, 1))
        self.assertTrue(sig_mng.is_valid(2, 6))
        sig_mng.add_signature(2, 6)

        # add signature to a singleton
        self.assertFalse(sig_mng.is_valid(1, 3))
        self.assertFalse(sig_mng.is_valid(1, 5))
        self.assertFalse(sig_mng.is_valid(1, 6))
        self.assertFalse(sig_mng.is_valid(4, 6))
        self.assertTrue(sig_mng.is_valid(1, 7))
        self.assertTrue(sig_mng.is_valid(4, 7))
        sig_mng.add_signature(1, 7)

        # remove the fraternal block
        sig_mng.remove_signature(2, 6)
        sig_mng.remove_signature(6, 5)
        sig_mng.remove_signature(0, 3)

        self.assertTrue(sig_mng.is_valid(4, 6))
        self.assertFalse(sig_mng.is_valid(4, 7))
        self.assertTrue(sig_mng.is_valid(4, 8))

        sig_mng.remove_signature(1, 7)
        self.assertTrue(sig_mng.is_valid(4, 7))

        # another test set
        block_ids = [0, -1, 0, 1, 2, 1, 0, 2, 1, -1, 1, -1]
        sig_mng = bswd_dw_lp.SignatureManager(block_ids)
        sig_mng.add_signature(1, 3)
        self.assertFalse(sig_mng.is_valid(0, 3))
        sig_mng.add_signature(2, 4)
        self.assertFalse(sig_mng.is_valid(0, 3))
        sig_mng.add_signature(6, 8)
        self.assertFalse(sig_mng.is_valid(0, 3))

    def test_remove_signature(self):
        """Tests remove_signature()."""

        block_ids = [-1, 0, 0, 1, 1]
        sig_mng = bswd_dw_lp.SignatureManager(block_ids)
        sig_mng.add_signature(0, 1)
        sig_mng.add_signature(1, 2)
        sig_mng.remove_signature(1, 2)

        self.assertDictEqual(sig_mng._signature_to_block_id, {1: -1})
        self.assertDictEqual(sig_mng._block_id_to_signatures, {})

        sig_mng.add_signature(1, 3)
        sig_mng.add_signature(3, 2)
        sig_mng.add_signature(2, 3)

        self.assertDictEqual(sig_mng._signature_to_block_id, {1: -1, 3: 0, 2: 1})
        self.assertDictEqual(sig_mng._block_id_to_signatures, {0: [3, 3], 1: [2]})

        sig_mng.remove_signature(2, 3)

        self.assertDictEqual(sig_mng._signature_to_block_id, {1: -1, 3: 0, 2: 1})
        self.assertDictEqual(sig_mng._block_id_to_signatures, {0: [3], 1: [2]})

        sig_mng.add_signature(2, 4)
        sig_mng.remove_signature(2, 4)

        self.assertDictEqual(sig_mng._signature_to_block_id, {1: -1, 3: 0, 2: 1})
        self.assertDictEqual(sig_mng._block_id_to_signatures, {0: [3], 1: [2]})

        sig_mng.add_signature(2, 5)
        sig_mng.remove_signature(2, 5)
        sig_mng.add_signature(2, 6)
        sig_mng.remove_signature(2, 6)

        self.assertDictEqual(sig_mng._signature_to_block_id, {1: -1, 3: 0, 2: 1})
        self.assertDictEqual(sig_mng._block_id_to_signatures, {0: [3], 1: [2]})

        sig_mng.remove_signature(3, 2)
        sig_mng.remove_signature(1, 3)

        self.assertDictEqual(sig_mng._signature_to_block_id, {1: -1})
        self.assertDictEqual(sig_mng._block_id_to_signatures, {})

        sig_mng.remove_signature(0, 1)

        self.assertDictEqual(sig_mng._signature_to_block_id, {})
        self.assertDictEqual(sig_mng._block_id_to_signatures, {})

    def test_is_valid_fraternal(self):
        block_ids = [0, 0, 0]
        sig_mng = bswd_dw_lp.SignatureManager(block_ids)

        self.assertFalse(sig_mng._is_valid_fraternal(1, 1))
        self.assertFalse(sig_mng._is_valid_fraternal(1, 3))
        self.assertFalse(sig_mng._is_valid_fraternal(3, 1))
        self.assertFalse(sig_mng._is_valid_fraternal(2, 3))
        self.assertFalse(sig_mng._is_valid_fraternal(3, 2))
        self.assertFalse(sig_mng._is_valid_fraternal(3, 7))
        self.assertFalse(sig_mng._is_valid_fraternal(7, 3))
        self.assertFalse(sig_mng._is_valid_fraternal(5, 4))
        self.assertFalse(sig_mng._is_valid_fraternal(4, 5))
        self.assertFalse(sig_mng._is_valid_fraternal(5, 5))
        self.assertFalse(sig_mng._is_valid_fraternal(27, 10))
        self.assertFalse(sig_mng._is_valid_fraternal(10, 27))

        self.assertTrue(sig_mng._is_valid_fraternal(1, 2))
        self.assertTrue(sig_mng._is_valid_fraternal(2, 1))
        self.assertTrue(sig_mng._is_valid_fraternal(3, 5))
        self.assertTrue(sig_mng._is_valid_fraternal(5, 3))
        self.assertTrue(sig_mng._is_valid_fraternal(27, 14))
        self.assertTrue(sig_mng._is_valid_fraternal(14, 27))

        self.assertFalse(sig_mng._is_valid_fraternal(1208925819614629174706176, 1208925819614629174706176))
        self.assertFalse(sig_mng._is_valid_fraternal(1208925819614629174706176, 1208925819614629174706177))
        self.assertFalse(sig_mng._is_valid_fraternal(1208925819614629174706177, 1208925819614629174706176))
        self.assertTrue(sig_mng._is_valid_fraternal(1208925819614629174706176, 2417851639229258349412352))
        self.assertTrue(sig_mng._is_valid_fraternal(2417851639229258349412352, 1208925819614629174706176))
        self.assertTrue(sig_mng._is_valid_fraternal(3626777458843887524118528, 6044629098073145873530880))
        self.assertTrue(sig_mng._is_valid_fraternal(6044629098073145873530880, 3626777458843887524118528))
