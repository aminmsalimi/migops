"""Basic CLI surface tests."""

import unittest

from migops.cli import build_parser


class TestCLI(unittest.TestCase):

    def test_top_level_commands_are_registered(self):
        parser = build_parser()

        help_text = parser.format_help()

        for command in (
            "status",
            "profiles",
            "users",
            "validate",
            "diff",
            "plan",
            "snapshot",
            "apply",
            "restore",
            "split",
            "create",
            "destroy",
            "mode",
            "gi",
            "ci",
        ):
            self.assertIn(
                command,
                help_text,
            )


if __name__ == "__main__":
    unittest.main()
