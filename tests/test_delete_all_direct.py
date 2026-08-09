"""Regression tests for direct native delete-all behavior."""

import unittest
from unittest.mock import patch

from migops.nvidia import NvidiaSmiError


class TestDirectDeleteAll(unittest.TestCase):

    @patch("migops.lifecycle.run_nvidia_smi")
    def test_delete_all_gi_does_not_list_instances_first(
        self,
        mock_run,
    ):
        from migops.lifecycle import _delete_all_mig_instances_direct

        mock_run.side_effect = [
            "Successfully destroyed compute instances",
            "Successfully destroyed GPU instances",
        ]

        result = _delete_all_mig_instances_direct(
            "0",
            dry_run=False,
        )

        self.assertEqual(result, 0)
        self.assertEqual(
            mock_run.call_args_list[0].args[0],
            ["mig", "-dci", "-i", "0"],
        )
        self.assertEqual(
            mock_run.call_args_list[1].args[0],
            ["mig", "-dgi", "-i", "0"],
        )

    @patch("migops.lifecycle.run_nvidia_smi")
    def test_empty_ci_does_not_prevent_gi_destroy(
        self,
        mock_run,
    ):
        from migops.lifecycle import _delete_all_mig_instances_direct

        mock_run.side_effect = [
            NvidiaSmiError(
                "No compute instances found: Not Found"
            ),
            "Successfully destroyed GPU instances",
        ]

        result = _delete_all_mig_instances_direct(
            "0",
            dry_run=False,
        )

        self.assertEqual(result, 0)
        self.assertEqual(mock_run.call_count, 2)

    @patch("migops.lifecycle.run_nvidia_smi")
    def test_permission_error_from_actual_destroy_is_not_hidden(
        self,
        mock_run,
    ):
        from migops.lifecycle import _delete_all_mig_instances_direct

        mock_run.side_effect = NvidiaSmiError(
            "No GPU instances found: Insufficient Permissions"
        )

        result = _delete_all_mig_instances_direct(
            "0",
            dry_run=False,
        )

        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
