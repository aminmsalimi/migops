"""Tests for MIG profile parsing."""

import unittest
from unittest.mock import patch

from migops.profiles import parse_profiles, query_profiles


H100_PROFILE_OUTPUT = """\
+-----------------------------------------------------------------------------+
| GPU instance profiles:                                                      |
| GPU   Name             ID    Instances   Memory     P2P    SM    DEC   ENC  |
|                              Free/Total   GiB              CE    JPEG  OFA  |
|=============================================================================|
|   0  MIG 1g.10gb       19     7/7        9.75       No     16     1     0   |
|                                                             1     1     0   |
+-----------------------------------------------------------------------------+
|   0  MIG 2g.20gb       14     3/3       19.50       No     32     2     0   |
|                                                             2     2     0   |
+-----------------------------------------------------------------------------+
|   0  MIG 3g.40gb        9     2/2       39.25       No     48     3     0   |
|                                                             3     3     0   |
+-----------------------------------------------------------------------------+
|   0  MIG 4g.40gb        5     1/1       39.25       No     64     4     0   |
|                                                             4     4     0   |
+-----------------------------------------------------------------------------+
|   0  MIG 7g.80gb        0     1/1       79.00       No    112     7     0   |
|                                                             8     7     1   |
+-----------------------------------------------------------------------------+
"""


class TestProfiles(unittest.TestCase):

    def test_parse_profiles(self):
        profiles = parse_profiles(H100_PROFILE_OUTPUT)

        self.assertEqual(len(profiles), 5)

        self.assertEqual(profiles[0].gpu, "0")
        self.assertEqual(profiles[0].name, "1g.10gb")
        self.assertEqual(profiles[0].profile_id, "19")
        self.assertEqual(profiles[0].free, 7)
        self.assertEqual(profiles[0].total, 7)

        self.assertEqual(profiles[2].name, "3g.40gb")
        self.assertEqual(profiles[2].free, 2)

        self.assertEqual(profiles[4].name, "7g.80gb")
        self.assertEqual(profiles[4].profile_id, "0")

    @patch("migops.profiles.run_nvidia_smi")
    def test_query_profiles_all_gpus(self, mock_run):
        mock_run.return_value = H100_PROFILE_OUTPUT

        profiles = query_profiles()

        mock_run.assert_called_once_with(
            ["mig", "-lgip"]
        )

        self.assertEqual(len(profiles), 5)

    @patch("migops.profiles.run_nvidia_smi")
    def test_query_profiles_specific_gpu(self, mock_run):
        mock_run.return_value = H100_PROFILE_OUTPUT

        query_profiles("0")

        mock_run.assert_called_once_with(
            ["mig", "-lgip", "-i", "0"]
        )


if __name__ == "__main__":
    unittest.main()