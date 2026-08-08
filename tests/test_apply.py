"""Tests for safe desired-state apply decisions."""

import unittest

from migops.apply import requires_workload_check


class TestApplySafety(unittest.TestCase):

    def test_enabling_mig_requires_workload_check(self):
        self.assertTrue(
            requires_workload_check(
                current_enabled=False,
                desired_enabled=True,
                current_profiles={},
                desired_profiles={
                    "3g.40gb": 1,
                },
            )
        )

    def test_disabling_mig_requires_workload_check(self):
        self.assertTrue(
            requires_workload_check(
                current_enabled=True,
                desired_enabled=False,
                current_profiles={
                    "3g.40gb": 1,
                },
                desired_profiles={},
            )
        )

    def test_replacing_layout_requires_workload_check(self):
        self.assertTrue(
            requires_workload_check(
                current_enabled=True,
                desired_enabled=True,
                current_profiles={
                    "1g.10gb": 4,
                },
                desired_profiles={
                    "3g.40gb": 1,
                },
            )
        )

    def test_matching_layout_does_not_require_check(self):
        self.assertFalse(
            requires_workload_check(
                current_enabled=True,
                desired_enabled=True,
                current_profiles={
                    "3g.40gb": 1,
                },
                desired_profiles={
                    "3g.40gb": 1,
                },
            )
        )


if __name__ == "__main__":
    unittest.main()
