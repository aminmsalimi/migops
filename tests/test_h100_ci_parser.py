"""Regression test for real H100 CI listing output."""

import unittest

from migops.workloads import parse_compute_instances


H100_LCI = r"""
+--------------------------------------------------------------------+
| Compute instances:                                                 |
| GPU     GPU       Name             Profile   Instance   Placement  |
|       Instance                       ID        ID       Start:Size |
|         ID                                                         |
|====================================================================|
|   0      6       MIG 1g.24gb          7         0          0:2     |
+--------------------------------------------------------------------+
"""


def _field(instance, *names):
    for name in names:
        if hasattr(instance, name):
            return getattr(instance, name)
    return None


class TestH100CIParser(unittest.TestCase):

    def test_real_h100_lci_row_is_detected(self):
        instances = parse_compute_instances(H100_LCI)

        self.assertEqual(len(instances), 1)
        instance = instances[0]

        gpu = _field(instance, "gpu", "gpu_id", "gpu_index")
        gi = _field(
            instance,
            "gi",
            "gi_id",
            "gpu_instance",
            "gpu_instance_id",
        )
        profile = _field(instance, "profile", "profile_name", "name")
        profile_id = _field(instance, "profile_id")
        ci = _field(
            instance,
            "ci",
            "ci_id",
            "instance",
            "instance_id",
            "compute_instance",
            "compute_instance_id",
        )
        placement = _field(instance, "placement")

        self.assertEqual(str(gpu), "0")
        self.assertEqual(str(gi), "6")
        self.assertEqual(str(profile), "1g.24gb")
        self.assertEqual(str(profile_id), "7")
        self.assertEqual(str(ci), "0")

        if placement is not None:
            self.assertEqual(str(placement), "0:2")


if __name__ == "__main__":
    unittest.main()
