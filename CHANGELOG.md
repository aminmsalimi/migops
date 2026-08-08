# Changelog\n\n## Unreleased\n\n### Changed\n\n- Simplified the public CLI to readable positional GPU syntax such as `migops split gpu 0 2` and first-class `migops enable gpu 0` / `migops disable gpu 0` commands.\n\n# Changelog

## Unreleased

### Fixed

- Fixed false MIG safety blocking when MIG mode is enabled with zero GPU/Compute Instances; NVIDIA `No ... instances found` responses are now treated as normal empty state.

### Changed

- Separated split recommendation from execution: `migops recommend gpu 0 2` is read-only, while `sudo migops split gpu 0 2` performs the split.\n
- Merged environment diagnostics into `migops status` and removed the separate `migops doctor` command.

### Added

- GPU and MIG status inspection
- MIG profile discovery
- workload discovery
- environment diagnostics
- Smart Split recommendations
- simple GI + CI creation
- safe CI-before-GI destruction
- MIG mode control
- advanced GI and CI management
- YAML desired-state configuration
- validate, diff, and plan workflows
- snapshots and restore
- safe apply with pre-change snapshot and final verification
- JSON output for automation-oriented commands
- GitHub Actions tests

### Safety

- workload checks before disruptive apply operations
- dry-run support
- explicit `--yes` requirement for real desired-state apply
