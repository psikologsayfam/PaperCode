"""Dependency and numerical smoke tests with synthetic data only."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODEL_PATH = Path(__file__).resolve().parents[1] / "NIV_HIRID_R6" / "fit_original46_ga_original_fitness.py"
SPEC = importlib.util.spec_from_file_location("hirid_ga", MODEL_PATH)
assert SPEC is not None and SPEC.loader is not None
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)


class ModelHelperTests(unittest.TestCase):
    def test_genetic_weights_are_normalised(self):
        rng = np.random.default_rng(2026)
        labels = np.repeat([0, 1], 25)
        scores = np.column_stack(
            [np.clip(0.20 + 0.55 * labels + rng.normal(0, 0.14, 50), 0.01, 0.99),
             np.clip(0.25 + 0.45 * labels + rng.normal(0, 0.16, 50), 0.01, 0.99)]
        )
        weights, history = MODEL.ga_weights(scores, labels, seed=7, pop=12, gens=5)
        self.assertEqual(weights.shape, (2,))
        self.assertTrue(np.all(weights >= 0))
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertEqual(len(history), 5)
        self.assertIn("delta", MODEL.delong(labels, scores[:, 0], scores[:, 1]))

    def test_training_pipeline_on_synthetic_examples(self):
        rng = np.random.default_rng(2026)
        features = rng.normal(size=(40, 5))
        labels = np.tile([0, 1], 20)
        features[3, 2] = np.nan
        clf = MODEL.pipeline("LR", seed=7)
        clf.fit(features, labels)
        self.assertEqual(clf.predict_proba(features).shape, (40, 2))


if __name__ == "__main__":
    unittest.main()
