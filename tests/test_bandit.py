import tempfile
import unittest
from pathlib import Path
import numpy as np
from purplepath.bandit import (chronological_split, generate, off_policy_evaluate,
                               rank, read_data, train)


class BanditTests(unittest.TestCase):
    def test_end_to_end_and_time_separation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = generate(root / "data.csv", 1000)
            days, actions, propensities, rewards, contexts = read_data(data)
            tr, va, te = chronological_split(days)
            self.assertLess(max(days[tr]), min(days[va]))
            self.assertLess(max(days[va]), min(days[te]))
            result = train(data, root / "model.json", root / "report.json")
            self.assertGreater(result["test_policy"]["matching_rows"], 0)
            suggestions = rank(root / "model.json", dict(zip(
                ("edr_coverage", "identity_coverage", "network_coverage",
                 "logging_maturity", "asset_criticality"), contexts[0])))
            self.assertEqual(len(suggestions), 5)
            self.assertGreaterEqual(suggestions[0]["priority_score"], suggestions[-1]["priority_score"])

    def test_known_ips_and_input_validation(self):
        result = off_policy_evaluate(np.array([0, 0, 0, 0]), np.array([0, 1, 0, 1]),
                                     np.full(4, .5), np.array([1, 0, 0, 1]))
        self.assertEqual(result["ips"], .5)
        self.assertEqual(result["snips"], .5)
        with tempfile.TemporaryDirectory() as d:
            data = generate(Path(d) / "data.csv", 100)
            data.write_text(data.read_text().replace("0.2,", "0,", 1))
            with self.assertRaises(ValueError):
                read_data(data)


if __name__ == "__main__":
    unittest.main()
