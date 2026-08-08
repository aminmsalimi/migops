"""Tests for MIGOps configuration parsing and validation."""

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


class TestConfigParsing(unittest.TestCase):

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
                            },
                            {
                                "profile": "1g.10gb",
                                "count": 2,
                            },
                        ],
                    }
                ],
            }
        )

        self.assertEqual(
            config.version,
            1,
        )

        self.assertEqual(
            len(config.gpus),
            1,
        )

        self.assertEqual(
            config.gpus[0].gpu,
            "0",
        )

        self.assertEqual(
            config.gpus[0].instances[1].count,
            2,
        )

    def test_reject_unknown_version(self):

        with self.assertRaises(
            ConfigError
        ):
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

    def test_reject_zero_count(self):

        with self.assertRaises(
            ConfigError
        ):
            parse_config_data(
                {
                    "version": 1,
                    "gpus": [
                        {
                            "gpu": "0",
                            "instances": [
                                {
                                    "profile": "1g.10gb",
                                    "count": 0,
                                }
                            ],
                        }
                    ],
                }
            )

    def test_reject_instances_when_mig_disabled(self):

        with self.assertRaises(
            ConfigError
        ):
            parse_config_data(
                {
                    "version": 1,
                    "gpus": [
                        {
                            "gpu": "0",
                            "mig_enabled": False,
                            "instances": [
                                {
                                    "profile": "1g.10gb",
                                    "count": 1,
                                }
                            ],
                        }
                    ],
                }
            )


class TestConfigValidation(unittest.TestCase):

    def setUp(self):

        self.gpu = GPU(
            index="0",
            name="NVIDIA H100 80GB HBM3",
            uuid="GPU-TEST",
            driver_version="580.00",
            pci_bus_id="00000000:31:00.0",
            mig_mode="Enabled",
        )

        self.profiles = [
            MigProfile(
                gpu="0",
                name="1g.10gb",
                profile_id="19",
                free=7,
                total=7,
                memory_gib=10.0,
            ),
            MigProfile(
                gpu="0",
                name="3g.40gb",
                profile_id="9",
                free=2,
                total=2,
                memory_gib=40.0,
            ),
        ]

    def test_valid_configuration(self):

        desired = GPUConfig(
            gpu="0",
            mig_enabled=True,
            instances=[
                ProfileRequest(
                    profile="3g.40gb",
                    count=1,
                )
            ],
        )

        result = validate_gpu_config(
            desired,
            self.gpu,
            self.profiles,
        )

        self.assertTrue(
            result.valid
        )

    def test_unsupported_profile_fails(self):

        desired = GPUConfig(
            gpu="0",
            mig_enabled=True,
            instances=[
                ProfileRequest(
                    profile="9g.999gb",
                    count=1,
                )
            ],
        )

        result = validate_gpu_config(
            desired,
            self.gpu,
            self.profiles,
        )

        self.assertFalse(
            result.valid
        )

    def test_too_many_instances_fails(self):

        desired = GPUConfig(
            gpu="0",
            mig_enabled=True,
            instances=[
                ProfileRequest(
                    profile="3g.40gb",
                    count=3,
                )
            ],
        )

        result = validate_gpu_config(
            desired,
            self.gpu,
            self.profiles,
        )

        self.assertFalse(
            result.valid
        )


if __name__ == "__main__":
    unittest.main()