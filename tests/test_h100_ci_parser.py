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


class TestH100CIParser(unittest.TestCase):

    def test_real_h100_lci_row_is_detected(self):
        instances = parse_compute_instances(H100_LCI)

        self.assertEqual(len(instances), 1)

        instance = instances[0]

        gpu = getattr(
            instance,
            "gpu",
            getattr(instance, "gpu_index", None),
        )
        gi = getattr(
            instance,
            "gi",
            getattr(instance, "gpu_instance_id", None),
        )
        ci = getattr(
            instance,
            "ci",
            getattr(instance, "compute_instance_id", None),
        )
        profile = getattr(
            instance,
            "profile",
            getattr(instance, "profile_name", None),
        )

        self.assertEqual(str(gpu), "0")
        self.assertEqual(str(gi), "6")
        self.assertEqual(str(ci), "0")
        self.assertEqual(profile, "1g.24gb")


if __name__ == "__main__":
    unittest.main()
