"""Tests for canonical MIG profile handling."""

import unittest

from migops.config import (
    GPUConfig,
    ProfileRequest,
    canonical_profile_counts,
)
from migops.profiles import MigProfile


class TestCanonicalProfiles(unittest.TestCase):

    def test_profile_id_becomes_profile_name(self):
        profiles = [
            MigProfile(
                gpu="0",
                name="3g.40gb",
                profile_id="9",
                free=2,
                total=2,
                memory_gib=40.0,
            )
        ]

        desired = GPUConfig(
            gpu="0",
            mig_enabled=True,
            instances=[
                ProfileRequest(
                    profile="9",
                    count=1,
                )
            ],
        )

        self.assertEqual(
            canonical_profile_counts(
                desired,
                profiles,
            ),
            {
                "3g.40gb": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
