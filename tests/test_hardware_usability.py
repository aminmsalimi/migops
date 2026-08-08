"""Hardware-usability regression tests."""

import unittest
from unittest.mock import patch

from migops.nvidia import NvidiaSmiError


class TestModeUsability(unittest.TestCase):

    @patch("migops.lifecycle.execute_operation")
    @patch("migops.lifecycle.get_matching_workloads")
    def test_mode_change_does_not_require_force_when_preflight_is_denied(
        self,
        mock_workloads,
        mock_execute,
    ):
        from migops.lifecycle import set_mig_mode

        mock_workloads.side_effect = NvidiaSmiError(
            "No GPU instances found: Insufficient Permissions"
        )
        mock_execute.return_value = 0

        result = set_mig_mode(
            "0",
            enabled=False,
            dry_run=False,
            force=False,
        )

        self.assertEqual(result, 0)
        mock_execute.assert_called_once()


class TestWorkloadFallback(unittest.TestCase):

    @patch("migops.workloads._query_workloads_fallback")
    @patch("migops.workloads._query_workloads_strict")
    def test_permission_failure_uses_fallback(
        self,
        mock_strict,
        mock_fallback,
    ):
        from migops.workloads import query_workloads

        marker = [object()]
        mock_strict.side_effect = NvidiaSmiError(
            "No GPU instances found: Insufficient Permissions"
        )
        mock_fallback.return_value = marker

        self.assertIs(
            query_workloads(),
            marker,
        )

    @patch("migops.workloads._query_workloads_strict")
    def test_real_driver_error_is_not_hidden(
        self,
        mock_strict,
    ):
        from migops.workloads import query_workloads

        mock_strict.side_effect = NvidiaSmiError(
            "GPU has fallen off the bus"
        )

        with self.assertRaises(NvidiaSmiError):
            query_workloads()


if __name__ == "__main__":
    unittest.main()
