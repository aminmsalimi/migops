"""Tests for MIGOps automatic MIG split planning."""

import unittest

from migops.profiles import MigProfile
from migops.split import (
    PhysicalGPU,
    recommend_split,
    select_gpu,
)


def profile(
    name: str,
    profile_id: str,
    memory: float,
    total: int,
) -> MigProfile:

    return MigProfile(
        gpu="0",
        name=name,
        profile_id=profile_id,
        free=total,
        total=total,
        memory_gib=memory,
    )


class TestSmartSplit(unittest.TestCase):

    def setUp(self):

        self.h100_96 = PhysicalGPU(
            index="0",
            name="NVIDIA H100 96GB HBM3",
            uuid="GPU-H100-96",
            pci_bus_id="00000000:31:00.0",
            memory_mib=96 * 1024,
            mig_mode="Enabled",
        )

        self.h100_96_profiles = [
            profile(
                "1g.12gb",
                "19",
                12.0,
                7,
            ),
            profile(
                "1g.24gb",
                "15",
                24.0,
                4,
            ),
            profile(
                "2g.24gb",
                "14",
                24.0,
                3,
            ),
            profile(
                "3g.48gb",
                "9",
                48.0,
                2,
            ),
            profile(
                "4g.48gb",
                "5",
                48.0,
                1,
            ),
            profile(
                "7g.96gb",
                "0",
                96.0,
                1,
            ),
        ]

    def test_split_h100_96_into_two(self):

        result = recommend_split(
            self.h100_96,
            self.h100_96_profiles,
            2,
        )

        self.assertEqual(
            result.profile,
            "3g.48gb",
        )

        self.assertEqual(
            result.requested_instances,
            2,
        )

        self.assertTrue(
            result.acceptable
        )

        self.assertEqual(
            result.memory_coverage_percent,
            100.0,
        )

    def test_split_h100_96_into_four(self):

        result = recommend_split(
            self.h100_96,
            self.h100_96_profiles,
            4,
        )

        self.assertEqual(
            result.profile,
            "1g.24gb",
        )

        self.assertTrue(
            result.acceptable
        )

        self.assertEqual(
            result.memory_coverage_percent,
            100.0,
        )

    def test_split_h100_80_into_four(self):

        gpu = PhysicalGPU(
            index="0",
            name="NVIDIA H100 80GB HBM3",
            uuid="GPU-H100-80",
            pci_bus_id="00000000:31:00.0",
            memory_mib=80 * 1024,
            mig_mode="Enabled",
        )

        profiles = [
            profile(
                "1g.10gb",
                "19",
                10.0,
                7,
            ),
            profile(
                "1g.20gb",
                "15",
                20.0,
                4,
            ),
            profile(
                "2g.20gb",
                "14",
                20.0,
                3,
            ),
            profile(
                "3g.40gb",
                "9",
                40.0,
                2,
            ),
            profile(
                "7g.80gb",
                "0",
                80.0,
                1,
            ),
        ]

        result = recommend_split(
            gpu,
            profiles,
            4,
        )

        self.assertEqual(
            result.profile,
            "1g.20gb",
        )

        self.assertTrue(
            result.acceptable
        )

    def test_three_way_split_warns_when_poor_fit(self):

        gpu = PhysicalGPU(
            index="0",
            name="NVIDIA H100 80GB HBM3",
            uuid="GPU-H100-80",
            pci_bus_id="00000000:31:00.0",
            memory_mib=80 * 1024,
            mig_mode="Enabled",
        )

        profiles = [
            profile(
                "1g.10gb",
                "19",
                10.0,
                7,
            ),
            profile(
                "2g.20gb",
                "14",
                20.0,
                3,
            ),
        ]

        result = recommend_split(
            gpu,
            profiles,
            3,
        )

        self.assertFalse(
            result.acceptable
        )

    def test_gpu_can_be_selected_by_uuid(self):

        selected = select_gpu(
            [self.h100_96],
            "GPU-H100-96",
        )

        self.assertEqual(
            selected.index,
            "0",
        )


if __name__ == "__main__":
    unittest.main()