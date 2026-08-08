"""Tests for MIG snapshot generation."""

import unittest
from unittest.mock import patch

from migops.lifecycle import GpuInstance
from migops.snapshot import snapshot_data
from migops.status import GPU


class TestSnapshot(unittest.TestCase):

    @patch(
        "migops.snapshot.query_gpu_instances"
    )
    @patch(
        "migops.snapshot.query_gpus"
    )
    def test_snapshot_groups_profiles(
        self,
        mock_gpus,
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

        mock_instances.return_value = [
            GpuInstance(
                "0",
                "1g.10gb",
                "19",
                "1",
                "0:1",
            ),
            GpuInstance(
                "0",
                "1g.10gb",
                "19",
                "2",
                "1:1",
            ),
        ]

        data = snapshot_data()

        self.assertEqual(
            data["gpus"][0]["instances"][0]["count"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
