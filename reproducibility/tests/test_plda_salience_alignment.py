from __future__ import annotations

import unittest

import numpy as np

from scripts.metrics.plda_salience_panels import distribution_alignment


class DistributionAlignmentTests(unittest.TestCase):
    def test_identical_distributions_have_unit_alignment(self) -> None:
        result = distribution_alignment(
            np.array([0.2, 0.3, 0.5]), np.array([0.2, 0.3, 0.5])
        )
        self.assertTrue(result["alignment_valid"])
        self.assertAlmostEqual(float(result["js_distance"]), 0.0)
        self.assertAlmostEqual(float(result["alignment_score"]), 1.0)

    def test_disjoint_distributions_have_zero_alignment(self) -> None:
        result = distribution_alignment(
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        )
        self.assertTrue(result["alignment_valid"])
        self.assertAlmostEqual(float(result["js_distance"]), 1.0)
        self.assertAlmostEqual(float(result["alignment_score"]), 0.0)

    def test_inputs_are_renormalized(self) -> None:
        result = distribution_alignment(
            np.array([2.0, 3.0, 5.0]), np.array([0.2, 0.3, 0.5])
        )
        self.assertTrue(result["alignment_valid"])
        self.assertAlmostEqual(float(result["alignment_score"]), 1.0)

    def test_zero_mass_is_invalid(self) -> None:
        result = distribution_alignment(
            np.array([0.2, 0.3, 0.5]), np.array([np.nan, np.nan, np.nan])
        )
        self.assertFalse(result["alignment_valid"])
        self.assertEqual(
            result["alignment_invalid_reason"], "zero_substantive_topic_mass"
        )

    def test_partial_nonfinite_vector_is_invalid(self) -> None:
        result = distribution_alignment(
            np.array([0.2, 0.3, 0.5]), np.array([0.2, np.nan, 0.8])
        )
        self.assertFalse(result["alignment_valid"])
        self.assertEqual(result["alignment_invalid_reason"], "nonfinite_topic_share")


if __name__ == "__main__":
    unittest.main()
