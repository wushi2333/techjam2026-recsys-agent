from __future__ import annotations

import unittest

from scripts.fault_matrix import run_matrix


class FaultMatrixTest(unittest.TestCase):
    def test_all_injected_faults_are_caught(self):
        rows = run_matrix()
        self.assertEqual(len(rows), 10)
        for row in rows:
            self.assertTrue(row["ok"], row)
