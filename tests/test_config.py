"""Tests for MIGOps desired-state configuration."""

import unittest

from migops.config import (
    ConfigError,
    GPUConfig,
    ProfileRequest,
    parse_config_data,
    validate_gpu_config,
)
from migops.profiles import MigProfile
from migops.status import GPU


class TestConfig(unittest.TestCase):

    def test_parse_valid_config(self):
        config = parse_config_data(
            {
                "version": 1,
                "gpus": [
                    {
                        "gpu": "0",
                        "mig_enabled": True,
                        "instances": [
                            {
                                "profile": "3g.40gb",
                                "count": 1,
                            }
                        ],
                    }
                ],
            }
        )

        self.assertEqual(
            config.gpus[0].instances[0].count,
            1,
        )

    def test_reject_unknown_version(self):
        with self.assertRaises(ConfigError):
            parse_config_data(
                {
                    "version": 2,
                    "gpus": [
                        {
                            "gpu": "0"
                        }
                    ],
                }
            )

    def test_validate_supported_profile(self):
        gpu = GPU(
            index="0",
            name="NVIDIA H100",
            uuid="GPU-TEST",
            driver_version="580.00",
            pci_bus_id="00000000:31:00.0",
            mig_mode="Enabled",
        )

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

        result = validate_gpu_config(
            GPUConfig(
                gpu="0",
                mig_enabled=True,
                instances=[
                    ProfileRequest(
                        profile="3g.40gb",
                        count=1,
                    )
                ],
            ),
            gpu,
            profiles,
        )

        self.assertTrue(result.valid)


if __name__ == "__main__":
    unittest.main()
