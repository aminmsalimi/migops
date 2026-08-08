"""Tests for MIG configuration drift detection."""

import unittest
from unittest.mock import patch

from migops.config import (
    GPUConfig,
    MigOpsConfig,
    ProfileRequest,
)
from migops.diffing import diff_config_object
from migops.lifecycle import GpuInstance
from migops.profiles import MigProfile
from migops.status import GPU


class TestDiff(unittest.TestCase):

    @patch(
        "migops.diffing.query_gpu_instances"
    )
    @patch(
        "migops.diffing.query_profiles"
    )
    @patch(
        "migops.diffing.query_gpus"
    )
    def test_matching_configuration(
        self,
        mock_gpus,
        mock_profiles,
        mock_instances,
    ):
        mock_gpus.return_value = [
            GPU(
                index="0",
                name="NVIDIA H100",
                uuid="GPU-TEST",
                driver_version="580",
                pci_bus_id="00000000:31:00.0",
                mig_mode="Enabled",
            )
        ]

        mock_profiles.return_value = [
            MigProfile(
                gpu="0",
                name="3g.40gb",
                profile_id="9",
                free=1,
                total=2,
                memory_gib=40.0,
            )
        ]

        mock_instances.return_value = [
            GpuInstance(
                gpu="0",
                profile="3g.40gb",
                profile_id="9",
                gi_id="1",
                placement="0:4",
            )
        ]

        result = diff_config_object(
            MigOpsConfig(
                version=1,
                gpus=[
                    GPUConfig(
                        gpu="0",
                        mig_enabled=True,
                        instances=[
                            ProfileRequest(
                                profile="3g.40gb",
                                count=1,
                            )
                        ],
                    )
                ],
            )
        )

        self.assertFalse(
            result.changed
        )


if __name__ == "__main__":
    unittest.main()