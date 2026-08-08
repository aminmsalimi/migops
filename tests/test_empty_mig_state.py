"""Regression tests for normal empty MIG states."""

import unittest
from unittest.mock import patch

from migops.lifecycle import (
    query_ci_instances,
    query_gpu_instances,
)
from migops.nvidia import NvidiaSmiError
from migops.workloads import query_compute_instances


class TestEmptyMigState(unittest.TestCase):

    @patch("migops.workloads.run_nvidia_smi")
    def test_workload_query_accepts_zero_compute_instances(
        self,
        mock_run,
    ):
        mock_run.side_effect = NvidiaSmiError(
            "No compute instances found: Not Found"
        )

        self.assertEqual(
            query_compute_instances(),
            [],
        )

    @patch("migops.lifecycle.run_nvidia_smi")
    def test_lifecycle_query_accepts_zero_gpu_instances(
        self,
        mock_run,
    ):
        mock_run.side_effect = NvidiaSmiError(
            "No GPU instances found: Not Found"
        )

        self.assertEqual(
            query_gpu_instances("0"),
            [],
        )

    @patch("migops.lifecycle.run_nvidia_smi")
    def test_lifecycle_query_accepts_zero_compute_instances(
        self,
        mock_run,
    ):
        mock_run.side_effect = NvidiaSmiError(
            "No compute instances found: Not Found"
        )

        self.assertEqual(
            query_ci_instances(
                gpu="0",
                gi="1",
            ),
            [],
        )

    @patch("migops.workloads.run_nvidia_smi")
    def test_real_nvidia_error_is_not_hidden(
        self,
        mock_run,
    ):
        mock_run.side_effect = NvidiaSmiError(
            "Insufficient Permissions"
        )

        with self.assertRaises(NvidiaSmiError):
            query_compute_instances()


if __name__ == "__main__":
    unittest.main()
