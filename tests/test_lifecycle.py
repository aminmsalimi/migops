"""Tests for MIG lifecycle management."""

import unittest

from migops.lifecycle import (
    build_ci_create_command,
    build_ci_delete_command,
    build_destroy_sequence,
    build_easy_create_command,
    build_gi_create_command,
    build_gi_delete_command,
    build_mode_command,
    parse_gpu_instances,
)


GI_OUTPUT = """\
+----------------------------------------------------+
| GPU instances:                                     |
| GPU   Name          Profile  Instance   Placement  |
|                       ID       ID       Start:Size |
|====================================================|
|   0  MIG 1g.10gb      19       13          6:1     |
+----------------------------------------------------+
|   0  MIG 3g.40gb       9        2          0:4     |
+----------------------------------------------------+
"""


class TestLifecycle(unittest.TestCase):

    def test_parse_gpu_instances(self):
        instances = parse_gpu_instances(
            GI_OUTPUT
        )

        self.assertEqual(
            len(instances),
            2,
        )

        self.assertEqual(
            instances[0].profile,
            "1g.10gb",
        )

        self.assertEqual(
            instances[0].gi_id,
            "13",
        )

        self.assertEqual(
            instances[1].profile,
            "3g.40gb",
        )

    def test_enable_mode_command(self):
        self.assertEqual(
            build_mode_command(
                "0",
                True,
            ),
            [
                "-i",
                "0",
                "-mig",
                "1",
            ],
        )

    def test_disable_mode_command(self):
        self.assertEqual(
            build_mode_command(
                "0",
                False,
            ),
            [
                "-i",
                "0",
                "-mig",
                "0",
            ],
        )

    def test_easy_create_one_complete_mig(self):
        self.assertEqual(
            build_easy_create_command(
                "0",
                "3g.40gb",
                1,
            ),
            [
                "mig",
                "-cgi",
                "3g.40gb",
                "-C",
                "-i",
                "0",
            ],
        )

    def test_easy_create_two_complete_migs(self):
        self.assertEqual(
            build_easy_create_command(
                "0",
                "3g.40gb",
                2,
            ),
            [
                "mig",
                "-cgi",
                "3g.40gb,3g.40gb",
                "-C",
                "-i",
                "0",
            ],
        )

    def test_create_gi_only(self):
        self.assertEqual(
            build_gi_create_command(
                "0",
                "3g.40gb",
            ),
            [
                "mig",
                "-cgi",
                "3g.40gb",
                "-i",
                "0",
            ],
        )

    def test_create_gi_with_ci(self):
        self.assertEqual(
            build_gi_create_command(
                "0",
                "3g.40gb",
                with_ci=True,
            ),
            [
                "mig",
                "-cgi",
                "3g.40gb",
                "-C",
                "-i",
                "0",
            ],
        )

    def test_create_ci_command(self):
        self.assertEqual(
            build_ci_create_command(
                "0",
                "2",
                "0",
            ),
            [
                "mig",
                "-cci",
                "0",
                "-gi",
                "2",
                "-i",
                "0",
            ],
        )

    def test_delete_specific_gi_command(self):
        self.assertEqual(
            build_gi_delete_command(
                "0",
                "2",
            ),
            [
                "mig",
                "-dgi",
                "-gi",
                "2",
                "-i",
                "0",
            ],
        )

    def test_delete_specific_ci_command(self):
        self.assertEqual(
            build_ci_delete_command(
                "0",
                gi="2",
                ci="0",
            ),
            [
                "mig",
                "-dci",
                "-ci",
                "0",
                "-gi",
                "2",
                "-i",
                "0",
            ],
        )

    def test_destroy_all_order(self):
        commands = build_destroy_sequence(
            "0"
        )

        self.assertEqual(
            commands,
            [
                [
                    "mig",
                    "-dci",
                    "-i",
                    "0",
                ],
                [
                    "mig",
                    "-dgi",
                    "-i",
                    "0",
                ],
            ],
        )

    def test_destroy_specific_gi_order(self):
        commands = build_destroy_sequence(
            "0",
            gi="2",
            ci_ids=["0", "1"],
        )

        self.assertEqual(
            commands,
            [
                [
                    "mig",
                    "-dci",
                    "-ci",
                    "0,1",
                    "-gi",
                    "2",
                    "-i",
                    "0",
                ],
                [
                    "mig",
                    "-dgi",
                    "-gi",
                    "2",
                    "-i",
                    "0",
                ],
            ],
        )


if __name__ == "__main__":
    unittest.main()