import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from tests.test_stage1_real_cache import _toy_joints


class Stage1ObservationDiagnosticTests(unittest.TestCase):
    def test_diagnose_observation_distribution_reports_abs_z_statistics(self):
        from Script.stage1.diagnose_observation_distribution import (
            diagnose_long_h5_observation_distribution,
        )

        class FakeAgent:
            obs_mean = torch.zeros(323, dtype=torch.float32)
            obs_std = torch.ones(323, dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            long_h5 = Path(tmpdir) / "long.h5"
            with h5py.File(long_h5, "w") as h5:
                group = h5.create_group("seq_000")
                group.create_dataset("joints_22", data=_toy_joints(length=12))

            summary = diagnose_long_h5_observation_distribution(
                long_h5_path=long_h5,
                agent=FakeAgent(),
                fps=20,
                max_sequences=1,
            )

        self.assertEqual(summary["converted_sequences"], 1)
        self.assertIn("aggregate_abs_z", summary)
        self.assertIn("p99", summary["aggregate_abs_z"])
        self.assertEqual(len(summary["worst_dimensions_by_p99_abs_z"]), 10)
        self.assertEqual(summary["sequences"][0]["sequence_id"], "seq_000")


if __name__ == "__main__":
    unittest.main()
