"""Tests for desired-state change planning."""

import unittest
from unittest.mock import patch

from migops.config import (
    GPUConfig,
    GPUValidationResult,
    MigOpsConfig,
    ProfileRequest,
    ValidationResult,
)
from migops.diffing import (
    DiffResult,
    GPUDiff,
)
from migops.planner import build_plan
from migops.status import GPU


class TestPlanner(unittest.TestCase):

    @patch(
        "migops.planner.query_workloads",
        return_value=[],
    )
    @patch(
        "migops.planner.query_gpus"
    )
    @patch(
        "migops.planner.diff_config_object"
    )
    @patch(
        "migops.planner.validate_config"
    )
    def test_reconfiguration_is_high_risk(
        self,
        mock_validate,
        mock_diff,
        mock_gpus,
        mock_workloads,
    ):
        mock_validate.return_value = ValidationResult(
            valid=True,
            config_version=1,
            gpu_results=[
                GPUValidationResult(
                    selector="0",
                    gpu_index="0",
                    gpu_name="NVIDIA H100",
                    valid=True,
                    requested_memory_gib=40,
                    messages=[],
                )
            ],
        )

        mock_diff.return_value = DiffResult(
            changed=True,
            gpus=[
                GPUDiff(
                    selector="0",
                    gpu_index="0",
                    gpu_name="NVIDIA H100",
                    desired_mig_enabled=True,
                    actual_mig_enabled=True,
                    desired_profiles={
                        "3g.40gb": 1
                    },
                    actual_profiles={
                        "1g.10gb": 4
                    },
                    changed=True,
                )
            ],
        )

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

        result = build_plan(
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

        self.assertEqual(
            result.gpus[0].risk,
            "HIGH",
        )


if __name__ == "__main__":
    unittest.main()
