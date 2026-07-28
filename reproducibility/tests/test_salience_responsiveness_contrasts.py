from __future__ import annotations

import unittest

from scripts.causality.recalculate_salience_responsiveness import (
    contrast_definitions,
)


class SalienceResponsivenessContrastTests(unittest.TestCase):
    def setUp(self) -> None:
        self.definitions = {
            item.short_name: item
            for item in contrast_definitions(
                mean_proximity=0.4, government_share=0.3
            )
        }

    def test_six_linkage_cells_are_reported(self) -> None:
        cells = [
            item for item in self.definitions.values() if item.role == "linkage_cell"
        ]
        self.assertEqual(len(cells), 6)

    def test_average_gap_respects_centered_proximity_coding(self) -> None:
        terms = self.definitions["h2a_average_government_gap"].terms
        self.assertEqual(terms["manifesto_x_government"], 1.0)
        self.assertAlmostEqual(
            terms["manifesto_x_proximity_centered_x_government"], -0.1
        )

    def test_average_cycle_change_uses_observed_government_share(self) -> None:
        terms = self.definitions["h1a_average_cycle_change"].terms
        self.assertEqual(terms["manifesto_x_proximity_centered"], 1.0)
        self.assertAlmostEqual(
            terms["manifesto_x_proximity_centered_x_government"], 0.3
        )

    def test_government_pre_election_linkage_uses_all_four_terms(self) -> None:
        terms = self.definitions["government_linkage_p1"].terms
        self.assertEqual(
            terms,
            {
                "manifesto_salience": 1.0,
                "manifesto_x_proximity_centered": 0.5,
                "manifesto_x_government": 1.0,
                "manifesto_x_proximity_centered_x_government": 0.5,
            },
        )


if __name__ == "__main__":
    unittest.main()
