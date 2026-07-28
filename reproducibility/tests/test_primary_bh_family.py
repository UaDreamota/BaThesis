from __future__ import annotations

import unittest

from scripts.causality.recalculate_primary_bh_family import apply_bh


class PrimaryBhFamilyTests(unittest.TestCase):
    def test_bh_uses_exactly_four_hypotheses(self) -> None:
        rows = [
            {"hypothesis": "H2b", "wild_p_two_sided": 0.16},
            {"hypothesis": "H1a", "wild_p_two_sided": 0.02},
            {"hypothesis": "H2a", "wild_p_two_sided": 0.13},
            {"hypothesis": "H1b", "wild_p_two_sided": 0.004},
        ]
        results = apply_bh(rows)
        self.assertEqual(
            results["hypothesis"].tolist(), ["H1a", "H1b", "H2a", "H2b"]
        )
        q = results.set_index("hypothesis")["bh_q_four_primary"]
        self.assertAlmostEqual(float(q["H1a"]), 0.04)
        self.assertAlmostEqual(float(q["H1b"]), 0.016)
        self.assertAlmostEqual(float(q["H2a"]), 0.16)
        self.assertAlmostEqual(float(q["H2b"]), 0.16)

    def test_missing_hypothesis_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly"):
            apply_bh(
                [
                    {"hypothesis": "H1a", "wild_p_two_sided": 0.02},
                    {"hypothesis": "H1b", "wild_p_two_sided": 0.01},
                    {"hypothesis": "H2a", "wild_p_two_sided": 0.20},
                ]
            )


if __name__ == "__main__":
    unittest.main()
