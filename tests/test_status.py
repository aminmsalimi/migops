"""Tests for MIGOps GPU and MIG status parsing."""

import unittest
from unittest.mock import patch

from migops.status import query_gpus, query_mig_devices


GPU_QUERY_OUTPUT = """\
0, NVIDIA H100 80GB HBM3, GPU-11111111-1111-1111-1111-111111111111, 580.105.08, 00000000:31:00.0, Enabled
1, NVIDIA H100 80GB HBM3, GPU-22222222-2222-2222-2222-222222222222, 580.105.08, 00000000:4B:00.0, Disabled
"""


MIG_LIST_OUTPUT = """\
GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-11111111-1111-1111-1111-111111111111)
  MIG 3g.40gb Device 0: (UUID: MIG-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa)
  MIG 2g.20gb Device 1: (UUID: MIG-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb)
  MIG 1g.10gb Device 2: (UUID: MIG-cccccccc-cccc-cccc-cccc-cccccccccccc)
GPU 1: NVIDIA H100 80GB HBM3 (UUID: GPU-22222222-2222-2222-2222-222222222222)
"""


class TestGPUQuery(unittest.TestCase):
    """Tests for GPU inventory parsing."""

    @patch("migops.status.run_nvidia_smi")
    def test_query_gpus(self, mock_run_nvidia_smi):
        mock_run_nvidia_smi.return_value = GPU_QUERY_OUTPUT

        gpus = query_gpus()

        self.assertEqual(len(gpus), 2)

        self.assertEqual(gpus[0].index, "0")
        self.assertEqual(gpus[0].name, "NVIDIA H100 80GB HBM3")
        self.assertEqual(
            gpus[0].uuid,
            "GPU-11111111-1111-1111-1111-111111111111",
        )
        self.assertEqual(gpus[0].driver_version, "580.105.08")
        self.assertEqual(gpus[0].pci_bus_id, "00000000:31:00.0")
        self.assertEqual(gpus[0].mig_mode, "Enabled")

        self.assertEqual(gpus[1].index, "1")
        self.assertEqual(gpus[1].mig_mode, "Disabled")

    @patch("migops.status.run_nvidia_smi")
    def test_query_mig_devices(self, mock_run_nvidia_smi):
        mock_run_nvidia_smi.return_value = MIG_LIST_OUTPUT

        devices = query_mig_devices()

        self.assertIn("0", devices)
        self.assertIn("1", devices)

        self.assertEqual(len(devices["0"]), 3)
        self.assertEqual(len(devices["1"]), 0)

        self.assertEqual(devices["0"][0].profile, "3g.40gb")
        self.assertEqual(devices["0"][0].device_id, "0")
        self.assertEqual(
            devices["0"][0].uuid,
            "MIG-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        )

        self.assertEqual(devices["0"][1].profile, "2g.20gb")
        self.assertEqual(devices["0"][2].profile, "1g.10gb")


if __name__ == "__main__":
    unittest.main()