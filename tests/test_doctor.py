"""Tests for MIGOps diagnostics."""

import unittest

from migops.status import mig_supported_from_mode


class TestDoctor(unittest.TestCase):

    def test_mig_enabled_is_supported(self):
        self.assertTrue(
            mig_supported_from_mode("Enabled")
        )

    def test_mig_disabled_is_supported(self):
        self.assertTrue(
            mig_supported_from_mode("Disabled")
        )

    def test_mig_na_is_not_supported(self):
        self.assertFalse(
            mig_supported_from_mode("N/A")
        )

    def test_unknown_state_is_not_supported(self):
        self.assertFalse(
            mig_supported_from_mode("Unknown")
        )


if __name__ == "__main__":
    unittest.main()