import unittest
import networkx as nx
import numpy as np

import algs.kernel as kernel
import algs.utils_misc as utils_misc


class TestKernel(unittest.TestCase):
    """Tests the kernel module."""

    def verify_adj_matrix(self, A, A_red, expected_blocks, k, v2):
        vs = A_red.index
        for u in vs:
            for v in vs:
                if u == v:
                    b = [block for block in expected_blocks if u in block][0]
                    self.assertGreaterEqual(len(b), 1)

                    if len(b) > (k if v2 else 2 ** k):
                        self.assertEqual(A_red[u][v], A[b[0]][b[1]])
                    else:
                        self.assertEqual(A_red[u][v], A[u][v])
                else:
                    self.assertEqual(A_red[u][v], A[u][v])

    def test_reduction_rules(self):
        """Tests reduction_rules()."""

        # tests vertex ordering
        G0 = nx.Graph()
        G0.add_node(4)
        G0.add_edges_from([(2, 3), (2, 0), (2, 1), (3, 0), (3, 1), (0, 1)], weight=2)
        G0.add_node(5)
        G0.add_node(6)
        A0 = nx.to_pandas_adjacency(G0, nodelist=[4, 2, 3, 0, 6, 1, 5], dtype=int)
        for i in range(len(G0)):
            A0.loc[i, i] = np.inf

        st = kernel.OrderingStrategy.ARBITRARY  # currently, it keeps a vertex with the smallest label
        A0r, removed, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A0, 3, kernel_v2_enabled=True, ordering_strategy=st)
        np.testing.assert_allclose(A0r.index.values, [4, 0, 6, 5])
        np.testing.assert_allclose(A0r.columns, [4, 0, 6, 5])
        self.assertEqual(removed, [1, 2, 3])
        self.assertEqual(num_total_blocks, 4)
        self.assertEqual(num_reduced_blocks, 1)

        st = kernel.OrderingStrategy.KEEP_FIRST
        A0r, removed, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A0, 3, kernel_v2_enabled=True, ordering_strategy=st)
        np.testing.assert_allclose(A0r.index.values, [4, 2, 6, 5])
        np.testing.assert_allclose(A0r.columns, [4, 2, 6, 5])
        self.assertEqual(removed, [0, 1, 3])
        self.assertEqual(num_total_blocks, 4)
        self.assertEqual(num_reduced_blocks, 1)

        st = kernel.OrderingStrategy.KEEP_LAST
        A0r, removed, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A0, 3, kernel_v2_enabled=True, ordering_strategy=st)
        np.testing.assert_allclose(A0r.index.values, [4, 6, 1, 5])
        np.testing.assert_allclose(A0r.columns, [4, 6, 1, 5])
        self.assertEqual(removed, [0, 2, 3])
        self.assertEqual(num_total_blocks, 4)
        self.assertEqual(num_reduced_blocks, 1)

        st = kernel.OrderingStrategy.PUSH_FRONT
        A0r, removed, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A0, 3, kernel_v2_enabled=True, ordering_strategy=st)
        np.testing.assert_allclose(A0r.index.values, [0, 4, 6, 5])
        np.testing.assert_allclose(A0r.columns, [0, 4, 6, 5])
        self.assertEqual(removed, [1, 2, 3])
        self.assertEqual(num_total_blocks, 4)
        self.assertEqual(num_reduced_blocks, 1)

        st = kernel.OrderingStrategy.PUSH_BACK
        A0r, removed, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A0, 3, kernel_v2_enabled=True, ordering_strategy=st)
        np.testing.assert_allclose(A0r.index.values, [4, 6, 5, 0])
        np.testing.assert_allclose(A0r.columns, [4, 6, 5, 0])
        self.assertEqual(removed, [1, 2, 3])
        self.assertEqual(num_total_blocks, 4)
        self.assertEqual(num_reduced_blocks, 1)

        # ------------------------------------------------------------
        #    1
        # ------------------------------------------------------------
        G1 = nx.read_weighted_edgelist('tests/resources/g_4_n428.txt', nodetype=int)
        A1 = utils_misc.get_wildcard_adjacency(G1)
        self.assertEqual(A1.shape[0], 428)

        expected_blocks = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53],
            [52],
            [384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383],
            [305],
            [112, 210, 227, 231],
            [54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 105, 106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 228, 229, 230, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255],
            [104],
            [286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355],
            [257],
            [256, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285]
        ]

        # kernel v1, no reordering
        st = kernel.OrderingStrategy.ARBITRARY
        k = 4
        A1r = A1.copy()
        A1r, removed, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A1r, k, kernel_v2_enabled=False, ordering_strategy=st)
        self.assertEqual(A1r.shape[0], 13)
        self.assertEqual(len(removed), 415)
        self.assertEqual(num_total_blocks, len(expected_blocks))
        self.assertEqual(num_reduced_blocks, 5)
        self.verify_adj_matrix(A1, A1r, expected_blocks, k, False)

        k = 5
        A1r = A1.copy()
        A1r, removed, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A1r, k, kernel_v2_enabled=False, ordering_strategy=st)
        self.assertEqual(A1r.shape[0], 41)
        self.assertEqual(len(removed), 387)
        self.assertEqual(num_total_blocks, len(expected_blocks))
        self.assertEqual(num_reduced_blocks, 4)
        self.verify_adj_matrix(A1, A1r, expected_blocks, k, False)

        k = 6
        A1r = A1.copy()
        A1r, removed, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A1r, k, kernel_v2_enabled=False, ordering_strategy=st)
        self.assertEqual(A1r.shape[0], 93)
        self.assertEqual(len(removed), 335)
        self.assertEqual(num_total_blocks, len(expected_blocks))
        self.assertEqual(num_reduced_blocks, 3)
        self.verify_adj_matrix(A1, A1r, expected_blocks, k, False)

        # kernel v1, with reordering
        st = kernel.OrderingStrategy.PUSH_FRONT
        k = 4
        A1r = A1.copy()
        A1r, removed, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A1r, k, kernel_v2_enabled=False, ordering_strategy=st)
        self.assertEqual(A1r.shape[0], 13)
        self.assertEqual(len(removed), 415)
        self.assertEqual(num_total_blocks, len(expected_blocks))
        self.assertEqual(num_reduced_blocks, 5)
        self.verify_adj_matrix(A1, A1r, expected_blocks, k, False)

        k = 5
        A1r = A1.copy()
        A1r, removed, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A1r, k, kernel_v2_enabled=False, ordering_strategy=st)
        self.assertEqual(A1r.shape[0], 41)
        self.assertEqual(len(removed), 387)
        self.assertEqual(num_total_blocks, len(expected_blocks))
        self.assertEqual(num_reduced_blocks, 4)
        self.verify_adj_matrix(A1, A1r, expected_blocks, k, False)

        k = 6
        A1r = A1.copy()
        A1r, removed, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A1r, k, kernel_v2_enabled=False, ordering_strategy=st)
        self.assertEqual(A1r.shape[0], 93)
        self.assertEqual(len(removed), 335)
        self.assertEqual(num_total_blocks, len(expected_blocks))
        self.assertEqual(num_reduced_blocks, 3)
        self.verify_adj_matrix(A1, A1r, expected_blocks, k, False)

        # kernel v2, no reordering
        st = kernel.OrderingStrategy.ARBITRARY
        for k in [4, 5, 6]:
            A1r = A1.copy()
            A1r, removed, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A1r, k, kernel_v2_enabled=True, ordering_strategy=st)
            self.assertEqual(A1r.shape[0], 13)
            self.assertEqual(len(removed), 415)
            self.assertEqual(num_total_blocks, len(expected_blocks))
            self.assertEqual(num_reduced_blocks, 5)
            self.verify_adj_matrix(A1, A1r, expected_blocks, k, True)

        # kernel v2, with reordering
        for st in [kernel.OrderingStrategy.KEEP_FIRST, kernel.OrderingStrategy.KEEP_LAST, kernel.OrderingStrategy.PUSH_FRONT, kernel.OrderingStrategy.PUSH_BACK]:
            for k in [4, 5, 6]:
                A1r = A1.copy()
                A1r, removed, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A1r, k, kernel_v2_enabled=True, ordering_strategy=st)
                self.assertEqual(A1r.shape[0], 13)
                self.assertEqual(len(removed), 415)
                self.assertEqual(num_total_blocks, len(expected_blocks))
                self.assertEqual(num_reduced_blocks, 5)
                self.verify_adj_matrix(A1, A1r, expected_blocks, k, True)

        # ------------------------------------------------------------
        #    2
        # ------------------------------------------------------------
        G2 = nx.read_weighted_edgelist('tests/resources/g_5_n271.txt', nodetype=int)
        A2 = utils_misc.get_wildcard_adjacency(G2)
        self.assertEqual(A2.shape[0], 271)

        expected_blocks = [
            [0, 1, 2, 3, 4, 6, 7],
            [5],
            [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 252, 253, 254, 255],
            [126],
            [8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40],
            [12, 39],
            [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201],
            [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
            [202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251]
        ]

        # kernel v1, no reordering
        st = kernel.OrderingStrategy.ARBITRARY
        k = 4
        A2r = A2.copy()
        A2r, removed, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A2r, k, kernel_v2_enabled=False, ordering_strategy=st)
        self.assertEqual(A2r.shape[0], 16)
        self.assertEqual(len(removed), 255)
        self.assertEqual(num_total_blocks, len(expected_blocks))
        self.assertEqual(num_reduced_blocks, 5)
        self.verify_adj_matrix(A2, A2r, expected_blocks, k, False)

        k = 5
        A2r = A2.copy()
        A2r, removed, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A2r, k, kernel_v2_enabled=False, ordering_strategy=st)
        self.assertEqual(A2r.shape[0], 93)
        self.assertEqual(len(removed), 178)
        self.assertEqual(num_total_blocks, len(expected_blocks))
        self.assertEqual(num_reduced_blocks, 2)
        self.verify_adj_matrix(A2, A2r, expected_blocks, k, False)

        k = 6
        A2r = A2.copy()
        A2r, removed, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A2r, k, kernel_v2_enabled=False, ordering_strategy=st)
        self.assertEqual(A2r.shape[0], 142)
        self.assertEqual(len(removed), 129)
        self.assertEqual(num_total_blocks, len(expected_blocks))
        self.assertEqual(num_reduced_blocks, 1)
        self.verify_adj_matrix(A2, A2r, expected_blocks, k, False)

        # kernel v1, with reordering
        st = kernel.OrderingStrategy.PUSH_FRONT
        k = 4
        A2r = A2.copy()
        A2r, removed, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A2r, k, kernel_v2_enabled=False, ordering_strategy=st)
        self.assertEqual(A2r.shape[0], 16)
        self.assertEqual(len(removed), 255)
        self.assertEqual(num_total_blocks, len(expected_blocks))
        self.assertEqual(num_reduced_blocks, 5)
        self.verify_adj_matrix(A2, A2r, expected_blocks, k, False)

        k = 5
        A2r = A2.copy()
        A2r, removed, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A2r, k, kernel_v2_enabled=False, ordering_strategy=st)
        self.assertEqual(A2r.shape[0], 93)
        self.assertEqual(len(removed), 178)
        self.assertEqual(num_total_blocks, len(expected_blocks))
        self.assertEqual(num_reduced_blocks, 2)
        self.verify_adj_matrix(A2, A2r, expected_blocks, k, False)

        k = 6
        A2r = A2.copy()
        A2r, removed, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A2r, k, kernel_v2_enabled=False, ordering_strategy=st)
        self.assertEqual(A2r.shape[0], 142)
        self.assertEqual(len(removed), 129)
        self.assertEqual(num_total_blocks, len(expected_blocks))
        self.assertEqual(num_reduced_blocks, 1)
        self.verify_adj_matrix(A2, A2r, expected_blocks, k, False)

        # kernel v2, no reordering
        st = kernel.OrderingStrategy.ARBITRARY
        for k in [4, 5, 6]:
            A2r = A2.copy()
            A2r, removed, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A2r, k, kernel_v2_enabled=True, ordering_strategy=st)
            self.assertEqual(A2r.shape[0], 10)
            self.assertEqual(len(removed), 261)
            self.assertEqual(num_total_blocks, len(expected_blocks))
            self.assertEqual(num_reduced_blocks, 6)
            self.verify_adj_matrix(A2, A2r, expected_blocks, k, True)

        # kernel v2, with reordering
        st = kernel.OrderingStrategy.PUSH_FRONT
        for k in [4, 5, 6]:
            A2r = A2.copy()
            A2r, removed, num_total_blocks, num_reduced_blocks = kernel.reduction_rules(A2r, k, kernel_v2_enabled=True, ordering_strategy=st)
            self.assertEqual(A2r.shape[0], 10)
            self.assertEqual(len(removed), 261)
            self.assertEqual(num_total_blocks, len(expected_blocks))
            self.assertEqual(num_reduced_blocks, 6)
            self.verify_adj_matrix(A2, A2r, expected_blocks, k, True)
