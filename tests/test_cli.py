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
            "split",
            "create",
            "destroy",
            "gi",
            "ci",
        ):
            self.assertIn(command, help_text)

        self.assertNotIn("doctor", help_text)
        self.assertNotIn("mode", help_text)

    def test_enable_syntax(self):
        args = self.parser.parse_args(
            ["enable", "gpu", "0"]
        )

        self.assertEqual(args.command, "enable")
        self.assertEqual(args.gpu, "0")

    def test_disable_syntax(self):
        args = self.parser.parse_args(
            ["disable", "gpu", "0"]
        )

        self.assertEqual(args.command, "disable")
        self.assertEqual(args.gpu, "0")

    def test_split_syntax(self):
        args = self.parser.parse_args(
            ["split", "gpu", "0", "2"]
        )

        self.assertEqual(args.gpu, "0")
        self.assertEqual(args.instances, 2)

    def test_split_dry_run_syntax(self):
        args = self.parser.parse_args(
            ["split", "gpu", "0", "4", "--dry-run"]
        )

        self.assertTrue(args.dry_run)

    def test_profiles_gpu_syntax(self):
        args = self.parser.parse_args(
            ["profiles", "gpu", "0"]
        )

        self.assertEqual(args.gpu, "0")

    def test_users_gpu_syntax(self):
        args = self.parser.parse_args(
            ["users", "gpu", "0"]
        )

        self.assertEqual(args.gpu, "0")

    def test_create_syntax(self):
        args = self.parser.parse_args(
            ["create", "gpu", "0", "3g.40gb"]
        )

        self.assertEqual(args.gpu, "0")
        self.assertEqual(args.profile, "3g.40gb")

    def test_destroy_syntax(self):
        args = self.parser.parse_args(
            ["destroy", "gpu", "0", "--all"]
        )

        self.assertTrue(args.all)

    def test_gi_syntax(self):
        args = self.parser.parse_args(
            ["gi", "create", "gpu", "0", "3g.40gb"]
        )

        self.assertEqual(args.gi_action, "create")
        self.assertEqual(args.profile, "3g.40gb")

    def test_ci_create_syntax(self):
        args = self.parser.parse_args(
            [
                "ci",
                "create",
                "gpu",
                "0",
                "gi",
                "1",
                "3g.40gb",
            ]
        )

        self.assertEqual(args.gpu, "0")
        self.assertEqual(args.gi, "1")
        self.assertEqual(args.profile, "3g.40gb")


if __name__ == "__main__":
    unittest.main()
