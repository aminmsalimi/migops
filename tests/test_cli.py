"""Tests for the public MIGOps CLI syntax."""

import unittest

from migops.cli import build_parser


class TestCLI(unittest.TestCase):

    def setUp(self):
        self.parser = build_parser()

    def test_top_level_commands_are_registered(self):
        help_text = self.parser.format_help()

        for command in (
            "status",
            "profiles",
            "users",
            "enable",
            "disable",
            "validate",
            "diff",
            "plan",
            "snapshot",
            "apply",
            "restore",
            "recommend",
            "split",
            "create",
            "destroy",
            "gi",
            "ci",
        ):
            self.assertIn(command, help_text)

        self.assertNotIn("doctor", help_text)
        self.assertNotIn("mode", help_text)

    def test_recommend_syntax(self):
        args = self.parser.parse_args(
            ["recommend", "gpu", "0", "2"]
        )

        self.assertEqual(args.command, "recommend")
        self.assertEqual(args.gpu, "0")
        self.assertEqual(args.instances, 2)

    def test_split_syntax(self):
        args = self.parser.parse_args(
            ["split", "gpu", "0", "2"]
        )

        self.assertEqual(args.command, "split")
        self.assertEqual(args.gpu, "0")
        self.assertEqual(args.instances, 2)

    def test_split_has_no_apply_flag(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(
                ["split", "gpu", "0", "2", "--apply"]
            )

    def test_split_dry_run_syntax(self):
        args = self.parser.parse_args(
            ["split", "gpu", "0", "4", "--dry-run"]
        )

        self.assertTrue(args.dry_run)

    def test_enable_syntax(self):
        args = self.parser.parse_args(
            ["enable", "gpu", "0"]
        )
        self.assertEqual(args.gpu, "0")

    def test_disable_syntax(self):
        args = self.parser.parse_args(
            ["disable", "gpu", "0"]
        )
        self.assertEqual(args.gpu, "0")


if __name__ == "__main__":
    unittest.main()
