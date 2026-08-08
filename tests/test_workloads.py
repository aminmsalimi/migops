"""Tests for NVIDIA GPU workload discovery."""

import unittest
from unittest.mock import patch

from migops.workloads import (
    parse_compute_instances,
    parse_memory,
    parse_processes,
    query_workloads,
)


MIG_PROCESS_OUTPUT = """\
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0    1    0      59785      C   python3                                      153MiB |
|    0    2    0      59885      C   /usr/bin/vllm                               8120MiB |
+-----------------------------------------------------------------------------------------+
"""


NON_MIG_PROCESS_OUTPUT = """\
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU        PID   Type   Process name                            GPU Memory |
|                                                                  Usage      |
|=============================================================================|
|    0       1234      C   python3                                    2048MiB |
+-----------------------------------------------------------------------------+
"""


CI_OUTPUT = """\
+-------------------------------------------------------+
| Compute instances:                                    |
| GPU     GPU       Name             Profile   Instance |
|       Instance                       ID        ID     |
|         ID                                            |
|=======================================================|
|   0      1       MIG 1c.3g.40gb       0         0     |
+-------------------------------------------------------+
|   0      2       MIG 2g.20gb          1         0     |
+-------------------------------------------------------+
"""


class TestWorkloads(unittest.TestCase):

    def test_parse_memory(self):
        self.assertEqual(parse_memory("153MiB"), 153)
        self.assertEqual(parse_memory("8120 MiB"), 8120)
        self.assertIsNone(parse_memory("N/A"))

    def test_parse_mig_processes(self):
        processes = parse_processes(MIG_PROCESS_OUTPUT)

        self.assertEqual(len(processes), 2)

        self.assertEqual(processes[0].gpu, "0")
        self.assertEqual(processes[0].gi_id, "1")
        self.assertEqual(processes[0].ci_id, "0")
        self.assertEqual(processes[0].pid, 59785)
        self.assertEqual(processes[0].process_name, "python3")
        self.assertEqual(processes[0].memory_mib, 153)

        self.assertEqual(processes[1].gi_id, "2")
        self.assertEqual(processes[1].memory_mib, 8120)

    def test_parse_non_mig_process(self):
        processes = parse_processes(NON_MIG_PROCESS_OUTPUT)

        self.assertEqual(len(processes), 1)

        process = processes[0]

        self.assertEqual(process.gpu, "0")
        self.assertIsNone(process.gi_id)
        self.assertIsNone(process.ci_id)
        self.assertEqual(process.pid, 1234)
        self.assertEqual(process.memory_mib, 2048)

    def test_parse_compute_instances(self):
        instances = parse_compute_instances(CI_OUTPUT)

        self.assertEqual(len(instances), 2)

        self.assertEqual(instances[0].gpu, "0")
        self.assertEqual(instances[0].gi_id, "1")
        self.assertEqual(instances[0].ci_id, "0")
        self.assertEqual(instances[0].profile, "1c.3g.40gb")

        self.assertEqual(instances[1].profile, "2g.20gb")

    @patch("migops.workloads.get_process_username")
    @patch("migops.workloads.query_compute_instances")
    @patch("migops.workloads.run_nvidia_smi")
    def test_query_workloads(
        self,
        mock_nvidia_smi,
        mock_compute_instances,
        mock_username,
    ):
        mock_nvidia_smi.return_value = MIG_PROCESS_OUTPUT

        mock_compute_instances.return_value = parse_compute_instances(
            CI_OUTPUT
        )

        mock_username.return_value = "amin"

        processes = query_workloads()

        self.assertEqual(len(processes), 2)

        self.assertEqual(processes[0].username, "amin")
        self.assertEqual(
            processes[0].mig_profile,
            "1c.3g.40gb",
        )

        self.assertEqual(
            processes[1].mig_profile,
            "2g.20gb",
        )


if __name__ == "__main__":
    unittest.main()